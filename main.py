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

# Import ReportLab modules for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

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
    
    # Initialize the model
    model = GenerativeModel(st.secrets["GCP_MODEL_CRED"]
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
   Below is a comprehensive prompt you can provide to a Large Language Model (LLM) that explains the task, input/output requirements, and the reasoning steps it should follow. Adapt the prompt as needed for your specific LLM environment.

---

**PROMPT START**

You are an AI assistant that is provided with a single image containing one or more FMCG (Fast Moving Consumer Goods) products. Your task is to analyze the image and produce a structured JSON output containing detailed information about each product identified in the image. Follow the instructions carefully and produce the response strictly in the required format.

### Input Description
- You will be given a single image of some common FMCG product(s).
- There can be multiple distinct products in the same image.
- A product may appear multiple times in the same image.

### What You Need To Identify
1. **Brand:**  
   - Identify the brand name of each distinct product visible in the image.  
   - There might be text or logo cues on the packaging that you can use to determine the brand name.  
   - If the brand is not recognizable or no brand is visible, output a best guess or "Unknown".

2. **Expiry Date:**  
   - Find and extract any expiry date information on the product packaging. The label might be indicated by phrases like "Expiry Date," "Best Before," "Use By," "BB," or similar.  
   - The expiry date might be given in various formats, such as:  
     - DD/MM/YYYY  
     - MM/YY  
     - DD-MM-YYYY  
     - Only a month and year (MM/YYYY)  
     - Only a year (YYYY)  
     - Relative information like "Best before 6 months from manufacture date" (if a manufacture date is also visible).  
   - **IMPORTANT** Convert all identified expiry date information into the format `DD/MM/YYYY` only.  
     - If the day (DD) is missing, use `01` as the default day.  
     - If the month (MM) is missing, use `01` (January) as the default month.  
     - If only a year (YYYY) is given, assume `01/01/YYYY`.  
     - If you need to compute a date from a given duration (e.g., "Best Before 6 months from a given manufacturing date"), add the specified duration to the manufacturing date and then format the resulting date as `DD/MM/YYYY`. If the computed day is unclear, choose the first day of the computed month. If no manufacturing day is given but a month/year is, assume the 1st of that month. If only a year is given, assume 01/01 of that year, then add the months.  
   
   Your goal is to present a single, best-guess expiry date in `DD/MM/YYYY` format.

3. **Item Count:**  
   - Count the number of identical units of each product visible in the image.  
   - Handle scenarios such as overlapping items, partially occluded items, or products placed side-by-side.  
   - Provide a best possible count based on the visible evidence in the image.

### Handling Multiple Products
- If the image contains multiple distinct products (e.g., Product A and Product B), you must output a JSON array containing one object per product.  
- Each object in the array will contain:  
  - `"brand"`: The detected brand name of the product.  
  - `"expiry_date"`: The expiry date of the product in `DD/MM/YYYY` format following the rules above. If no expiry date is visible or cannot be inferred, return a sensible default such as `"01/01/2099"` (or another clearly invalid date placeholder if instructed).  
  - `"count"`: The integer count of how many units of that product are visible in the image.

### Output Format
- Your final answer **must** be a JSON array of objects, where each object has the keys `brand`, `expiry_date`, and `count`.  
- Example structure (pseudocode):
  ```json
  [
    {
      "brand": "Product brand name",
      "expiry_date": "DD/MM/YYYY",
      "count": 2
    },
    {
      "brand": "Second product brand name",
      "expiry_date": "DD/MM/YYYY",
      "count": 1
    }
    // ... and so on for other products if they exist
  ]
  ```

### Sample Output (For Illustration)
```json
[
  {
    "brand": "Nestle",
    "expiry_date": "01/12/2024",
    "count": 2
  },
  {
    "brand": "Cadbury",
    "expiry_date": "01/06/2025",
    "count": 1
  }
]
```

### Important Notes
- The date format is strict: Always `DD/MM/YYYY`. For missing components, make sensible assumptions as described.
- If the product’s expiry date references a "Best Before" period from a manufacturing date, COMPUTE THE expected date as best as possible.
- If no expiry date is found at all, provide a fallback date like `"NA"` to indicate no available data.
- Ensure the count accurately reflects the number of identical items of that product in the image.
- If there are multiple products, output one JSON object per product, all contained in a JSON array.

Now, using these instructions, analyze the given image and produce the final JSON output.

**PROMPT END**

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
        print(response.text)
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

                # Parse expiry date with various formats
                expiry_date_str = data['expiry_date'].strip()
                expiry_date = None

                # List of date formats to try
                date_formats = ["%d/%m/%Y", "%d/%m/%y", "%m/%Y", "%m/%y", "%Y", "%y"]

                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(expiry_date_str, fmt)

                        # If day is missing, default to 1
                        if '%d' not in fmt:
                            parsed_date = parsed_date.replace(day=1)

                        expiry_date = parsed_date
                        break  # Exit the loop if parsing is successful
                    except ValueError:
                        continue

                current_date = datetime.now()

                if expiry_date:
                    # Compare dates based on available components
                    if expiry_date.year < current_date.year:
                        is_expired = "Yes"
                    elif expiry_date.year == current_date.year:
                        if expiry_date.month < current_date.month:
                            is_expired = "Yes"
                        elif expiry_date.month == current_date.month:
                            if expiry_date.day < current_date.day:
                                is_expired = "Yes"
                            else:
                                is_expired = "No"
                        else:
                            is_expired = "No"
                    else:
                        is_expired = "No"

                    lifespan_days = (expiry_date - current_date).days
                    lifespan = lifespan_days if lifespan_days >= 0 else "NA"
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

            except Exception as e:
                st.error(f"Error processing product details: {str(e)}")
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

def generate_pdf_report(data, filename):
    try:
        doc = SimpleDocTemplate(filename, pagesize=letter)
        elements = []
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Title
        title = Paragraph("FMCG Product Analysis Report", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 20))  # Add more space after title
        
        # Report generation date
        date_text = Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
        elements.append(date_text)
        elements.append(Spacer(1, 12))
        
        # Convert data to table format
        table_data = [["Sl No", "Timestamp", "Brand", "Expiry Date", "Count", "Expired", "Expected Lifespan (Days)"]]
        for item in data:
            row = [
                str(item.get("Sl No", "")),
                item.get("Timestamp", "").split('T')[0],  # Format timestamp to show only date
                item.get("Brand", ""),
                item.get("Expiry Date", ""),
                str(item.get("Count", "")),
                item.get("Expired", ""),
                str(item.get("Expected Lifespan (Days)", ""))
            ]
            table_data.append(row)
        
        # Create and style table
        table = Table(table_data, repeatRows=1)  # Repeat header row on each page
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BOX', (0, 0), (-1, -1), 2, colors.black),
            ('LINEBELOW', (0, 0), (-1, 0), 2, colors.black),
        ]))
        
        elements.append(table)
        doc.build(elements)
        return True
        
    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")
        return False

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
                        # st.subheader("Raw Model Output:")
                        # st.code(analysis, language="json")

                        products = parse_product_details(analysis)
                        if products:  # Valid products list
                            update_product_data(products)
                            st.subheader("Product Details:")
                            for product in products:
                                # Display product details
                                st.markdown(f"**Product Details for {product['Brand']} (Expiry: {product['Expiry Date']}):**")
                                display_fields = {k:v for k,v in product.items() if k not in ['Sl No', 'Timestamp']}
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

        # Add option to generate report
        if st.button("Generate PDF Report"):
            # Generate report
            report_generated = generate_pdf_report(st.session_state.product_data, "product_report.pdf")
            if report_generated:
                st.success("Report generated successfully!")
                with open("product_report.pdf", "rb") as file:
                    btn = st.download_button(
                        label="Download PDF",
                        data=file,
                        file_name="product_report.pdf",
                        mime="application/octet-stream"
                    )
            else:
                st.error("Failed to generate report.")
    else:
        st.write("No products scanned yet.")

if __name__ == "__main__":
    main()
