import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import pandas as pd
from PIL import Image
import io
import json
import re
from google.oauth2 import service_account
from google.cloud import aiplatform
from datetime import datetime

# Initialize Streamlit page configuration
st.set_page_config(page_title="FMCG Product Analyzer", layout="wide")

# Load Google Cloud credentials
try:
    credentials_info = json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS"])
    credentials = service_account.Credentials.from_service_account_info(credentials_info)
    project_id = st.secrets["GOOGLE_CLOUD_PROJECT"]
    
    # Initialize Vertex AI
    vertexai.init(project=project_id, location="us-central1", credentials=credentials)
    
    # Initialize the Gemini model
    model = GenerativeModel("gemini-1.5-flash-002")
    st.success("Model loaded successfully")

except Exception as e:
    st.error(f"Error loading Google Cloud credentials: {str(e)}")
    st.stop()

# Initialize session state for product tracking
if 'product_data' not in st.session_state:
    st.session_state.product_data = []

def analyze_image(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    image_part = Part.from_data(img_byte_arr, mime_type="image/png")

    prompt = """
    Analyze this FMCG product image and provide the following details for each product detected in JSON format:

    [
        {
            "brand": "Product brand name",
            "expiry_date": "DD/MM/YYYY format. Extract from labels like 'Expiry Date', 'Best Before', 'Use By'",
            "count": "Number of identical items visible in the image, accounting for overlapping and partial views"
        },
        {
            // Next product
        }
        // Add more products as detected
    ]

    Requirements:
    1. Brand Detection:
       - Identify brand names even with partial visibility.
       - Handle multiple product instances of different brands.

     2. Expiry Date:
       - Support formats: DD/MM/YYYY, MM/YY, YYYY-MM-DD.
       - Convert all dates to DD/MM/YYYY format.
       - Look for text patterns: "Expiry Date:", "Best Before:" [if best before (in months is given then calculate it)], "Use By:".
         **Instructions for Expiry Date Formatting:**

    - **Acceptable Input Formats:**
      - **DD/MM/YYYY**
      - **MM/YYYY** (e.g., 12/2025)
      - **MM/YY** (e.g., 12/25)
      - **YYYY-MM-DD**
      - **DD-MM-YYYY**
      - **DD.MM.YYYY**
      - **Month Name Year** (e.g., December 2025)

    - Conversion Rules
      - If day is missing: Use `01` as the default day.
        - **Example:** `12/2025` ➔ `01/12/2025`
      - For two-digit years: Assume `20YY`.
        - **Example:** `12/25` ➔ `01/12/2025`
      - Convert month names to numeric format
        - **Example:** `December 2025` ➔ `01/12/2025`

    -Always return dates in `DD/MM/YYYY` format.

    3. Item Count:
       - Count identical products even when overlapping.
       - Account for side-by-side placement.
       - Include partially visible items.

    Return only the JSON array of product objects without any additional text or formatting.
    """

    try:
        response = model.generate_content(
            [image_part, prompt],
            generation_config={
                "max_output_tokens": 2048,
                "temperature": 0.1,
                "top_p": 1,
                "top_k": 32
            }
        )
        # Clean the response text
        response_text = response.text.strip()
        # Remove any potential markdown code block markers
        response_text = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.MULTILINE)
        return response_text
    except Exception as e:
        st.error(f"Error in image analysis: {str(e)}")
        return None

def parse_product_details(analysis):
    try:
        if not analysis or not isinstance(analysis, str):
            st.error("Invalid analysis response")
            return None

        # Clean the input string
        analysis = analysis.strip()

        try:
            products = json.loads(analysis)
            if not isinstance(products, list):
                st.error("Expected a list of products in the analysis")
                return None
        except json.JSONDecodeError as e:
            st.error(f"JSON parsing error: {str(e)}\nReceived text: {analysis}")
            return None

        parsed_products = []

        for data in products:
            # Validate required fields and their types
            required_fields = {
                'brand': str,
                'expiry_date': str,
                'count': (int, float, str)  # Allow multiple numeric types
            }

            for field, field_type in required_fields.items():
                if field not in data:
                    st.error(f"Missing required field '{field}' in one of the products")
                    return None
                if not isinstance(data[field], field_type):
                    if field == 'count':
                        try:
                            data[field] = int(float(data[field]))
                        except (ValueError, TypeError):
                            st.error(f"Invalid type for field '{field}': expected numeric, got {type(data[field])}")
                            return None
                    else:
                        st.error(f"Invalid type for field '{field}': expected {field_type}, got {type(data[field])}")
                        return None

            try:
                # Add current timestamp in ISO format with timezone
                current_timestamp = datetime.now().astimezone().isoformat()

                # Parse expiry date
                expiry_date = datetime.strptime(data['expiry_date'], "%d/%m/%Y")
                current_date = datetime.now()

                if expiry_date < current_date:
                    is_expired = "Yes"
                    lifespan = "NA"
                elif expiry_date > current_date:
                    is_expired = "No"
                    lifespan = (expiry_date - current_date).days
                else:
                    is_expired = "NA"
                    lifespan = "NA"

                parsed_product = {
                    "Sl No": None,  # Will be assigned later
                    "Timestamp": current_timestamp,
                    "Brand": data['brand'].strip(),
                    "Expiry Date": data['expiry_date'],
                    "Count": int(data['count']),
                    "Expired": is_expired,
                    "Expected Lifespan (Days)": lifespan
                }

                parsed_products.append(parsed_product)

            except ValueError as e:
                st.error(f"Date parsing error for one of the products: {str(e)}")
                return None

        return parsed_products

    except Exception as e:
        st.error(f"Error parsing product details: {str(e)}")
        return None

def update_product_data(products):
    if not products:
        return

    for product in products:
        # Check if a product with the same brand and expiry date exists
        existing_product = next(
            (p for p in st.session_state.product_data if p['Brand'] == product['Brand'] and p['Expiry Date'] == product['Expiry Date']),
            None
        )

        if existing_product:
            # Increase the count and update timestamp
            existing_product['Count'] += product['Count']
            existing_product['Timestamp'] = product['Timestamp']
        else:
            # Assign Sl No
            product["Sl No"] = len(st.session_state.product_data) + 1
            st.session_state.product_data.append(product)

def main():
    st.title("FMCG Product Analyzer and Tracker")
    
    uploaded_file = st.file_uploader("Choose an image of FMCG products", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Resize image for display
        max_width = 300
        ratio = max_width / image.width
        new_size = (max_width, int(image.height * ratio))
        resized_image = image.resize(new_size)
        
        # Display resized image
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(resized_image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            if st.button("Analyze Image"):
                with st.spinner("Analyzing image..."):
                    analysis = analyze_image(image)
                    if analysis:
                        products = parse_product_details(analysis)
                        if products:  # Valid products list
                            update_product_data(products)
                            st.subheader("Product Details:")
                            for product in products:
                                # Display product details
                                st.markdown(f"**Product Details for {product['Brand']} (Expiry: {product['Expiry Date']}):**")
                                display_fields = {k:v for k,v in product.items() if k not in ['Count', 'Sl No', 'Timestamp']}
                                for key, value in display_fields.items():
                                    st.write(f"**{key}:** {value}")
                                st.write("---")
                        else:
                            st.error("Failed to parse product details. Please try again.")
                    else:
                        st.error("Unable to analyze the image. Please try again with a different image.")
    
    st.subheader("Product Inventory")
    if st.session_state.product_data:
        df = pd.DataFrame(st.session_state.product_data)
        
        # Reorder columns
        columns_order = [
            'Sl No', 'Timestamp', 'Brand', 'Expiry Date', 
            'Count', 'Expired', 'Expected Lifespan (Days)'
        ]
        df = df[columns_order]
        
        # Style the dataframe
        styled_df = df.style.set_properties(**{
            'text-align': 'left',
            'white-space': 'nowrap'
        })
        styled_df = styled_df.set_table_styles([
            {'selector': 'th', 'props': [
                ('font-weight', 'bold'),
                ('text-align', 'left'),
                ('padding', '8px')
            ]},
            {'selector': 'td', 'props': [
                ('padding', '8px'),
                ('border', '1px solid #ddd')
            ]}
        ])
        
        st.write(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.write("No products scanned yet.")

if __name__ == "__main__":
    main()
