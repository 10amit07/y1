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


hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {display: none !important;}
        header {visibility: hidden;}
        .viewerBadge_container__1QSob {display: none !important;}
        .css-1lsmgbg.egzxvld1 {display: none !important;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


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
    The image can consist of multiple products. For every product in the image, do the following:
    Analyze this FMCG product image and provide the following details for each product detected in JSON format:

    [
        {
            "brand": "Product brand name",
            "expiry_date": "DD/MM/YYYY format. Extract from labels like 'Expiry Date', 'Best Before', 'Use By'",
            "count": "Number of identical items visible in the image, accounting for overlapping and partial views"
        },
        {
            "brand": "2nd product (IF IT EXISTS)",
            "expiry_date": "DD/MM/YYYY format. Extract from labels like 'Expiry Date', 'Best Before', 'Use By'",
            "count": "Number of identical items visible in the image, accounting for overlapping and partial views"
        }
    ]

    Requirements:
    1. Brand Detection:
       - Identify brand names even with partial visibility.
       - Handle multiple product instances of different brands.

    2. Expiry Date:
       - Support formats: DD/MM/YYYY, MM/YYYY, MM/YY, YYYY-MM-DD, DD-MM-YYYY, DD.MM.YYYY, MonthName Year.
       - CONVERT ALL THE DATES TO DD/MM/YYYY FORMAT ONLY. If no day is given, use 01 for day.
       - For two-digit years, assume 20YY.
       - Always return dates in DD/MM/YYYY format.

    3. Item Count:
       - Count identical products even when overlapping.
       - Include partially visible items.

    Return only the JSON array without additional text.
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
        print(response.text)
        response_text = response.text.strip()
        # Remove any potential markdown code block markers
        response_text = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.MULTILINE)
        return response_text
    except Exception as e:
        st.error(f"Error in image analysis: {str(e)}")
        return None

def normalize_date(date_str):
    """
    Normalize various date formats into DD/MM/YYYY format.
    Rules:
    - If day is missing, default to 01.
    - If year is two digits, assume 20YY.
    - Convert month names to numbers.
    - Acceptable inputs (examples):
      - DD/MM/YYYY
      - MM/YYYY
      - MM/YY
      - YYYY-MM-DD
      - DD-MM-YYYY
      - DD.MM.YYYY
      - MonthName YYYY (e.g., December 2025)
    """

    date_str = date_str.strip()

    # Try direct DD/MM/YYYY first
    try:
        dt = datetime.strptime(date_str, "%d/%m/%Y")
        return dt.strftime("%d/%m/%Y")
    except:
        pass

    # If YYYY-MM-DD
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%d/%m/%Y")
    except:
        pass

    # If DD-MM-YYYY
    try:
        dt = datetime.strptime(date_str, "%d-%m-%Y")
        return dt.strftime("%d/%m/%Y")
    except:
        pass

    # If DD.MM.YYYY
    try:
        dt = datetime.strptime(date_str, "%d.%m.%Y")
        return dt.strftime("%d/%m/%Y")
    except:
        pass

    # Check if format is MM/YYYY or MM/YY (no day given)
    # For MM/YYYY (e.g., 12/2023), default day to 01
    # For MM/YY (e.g., 12/23), assume year 20YY and default day to 01
    # We'll identify if it's something like "MM/YYYY" or "MM/YY" by splitting.
    slash_parts = date_str.split('/')
    if len(slash_parts) == 2:
        # Could be MM/YYYY or MM/YY
        mm, yy = slash_parts
        mm = mm.strip()
        yy = yy.strip()
        # Validate month
        if mm.isdigit():
            month = int(mm)
            if 1 <= month <= 12:
                # Check year length
                if len(yy) == 2:
                    # Two-digit year
                    year = int("20" + yy)
                else:
                    year = int(yy)

                # Default day is 01
                try:
                    dt = datetime(year, month, 1)
                    return dt.strftime("%d/%m/%Y")
                except:
                    pass

    # Check for month name and year (e.g., December 2025)
    # Extract month name and year
    # We look for a pattern: MonthName Year
    words = date_str.split()
    if len(words) == 2:
        month_name, year_str = words
        month_name = month_name.capitalize()
        if year_str.isdigit():
            year = int(year_str)
            # Convert month name to month number
            try:
                month_num = list(calendar.month_name).index(month_name)
                dt = datetime(year, month_num, 1)
                return dt.strftime("%d/%m/%Y")
            except:
                pass

    # If we got here, date might be in another format. Try some heuristic:
    # For example, MM/YY with '.' or '-' separators or something else.
    # We'll try a few more common patterns.

    # MM/YY (like 12/23)
    # Already handled above with slash check, but try again with a different approach
    # Just in case format is like "12/23" but didn't parse due to exceptions
    slash_parts = date_str.split('/')
    if len(slash_parts) == 2:
        mm, yy = slash_parts
        if mm.isdigit() and yy.isdigit():
            month = int(mm)
            if len(yy) == 2:
                year = int("20" + yy)
                try:
                    dt = datetime(year, month, 1)
                    return dt.strftime("%d/%m/%Y")
                except:
                    pass

    # If none of the above worked, raise an exception
    raise ValueError(f"Unrecognized date format: {date_str}")




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
                # Add current timestamp
                current_timestamp = datetime.now().astimezone().isoformat()

                # Normalize and parse the expiry_date
                normalized_date = normalize_date(data['expiry_date'])
                expiry_date = datetime.strptime(normalized_date, "%d/%m/%Y")
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
                    "Expiry Date": normalized_date,
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
