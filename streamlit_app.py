import streamlit as st
import openai
from PIL import Image
import pandas as pd
import json
import base64
import io

# Configure page
st.set_page_config(page_title="Invoice Analyzer", page_icon="📄", layout="wide")

# Initialize OpenAI client
@st.cache_resource
def init_openai():
    return openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def encode_image(image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def extract_invoice_data(image):
    """Extract invoice data using OpenAI Vision API"""
    
    try:
        client = init_openai()
        base64_image = encode_image(image)
        
        prompt = """
        Analyze this invoice image and extract the following information in JSON format:
        
        {
            "invoice_number": "",
            "invoice_date": "",
            "due_date": "",
            "vendor_name": "",
            "vendor_address": "",
            "vendor_tax_id": "",
            "customer_name": "",
            "customer_address": "",
            "line_items": [
                {
                    "description": "",
                    "quantity": "",
                    "unit_price": "",
                    "total_price": ""
                }
            ],
            "subtotal": "",
            "tax_amount": "",
            "total_amount": "",
            "currency": ""
        }
        
        If any field is not found, use "N/A" as the value.
        Return only the JSON object, no additional text.
        """
        
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        # Extract JSON from response
        content = response.choices[0].message.content
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        return json.loads(content)
        
    except Exception as e:
        st.error(f"Error extracting data: {str(e)}")
        return None

def display_extracted_data(data):
    """Display extracted data in organized format"""
    
    # Basic Invoice Info
    st.subheader("📋 Invoice Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Invoice Number", data.get("invoice_number", "N/A"))
    with col2:
        st.metric("Invoice Date", data.get("invoice_date", "N/A"))
    with col3:
        st.metric("Due Date", data.get("due_date", "N/A"))
    
    # Vendor & Customer Info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏢 Vendor Information")
        st.write(f"**Name:** {data.get('vendor_name', 'N/A')}")
        st.write(f"**Address:** {data.get('vendor_address', 'N/A')}")
        st.write(f"**Tax ID:** {data.get('vendor_tax_id', 'N/A')}")
    
    with col2:
        st.subheader("👤 Customer Information")
        st.write(f"**Name:** {data.get('customer_name', 'N/A')}")
        st.write(f"**Address:** {data.get('customer_address', 'N/A')}")
    
    # Line Items
    st.subheader("📦 Line Items")
    if data.get("line_items") and len(data["line_items"]) > 0:
        df = pd.DataFrame(data["line_items"])
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No line items found")
    
    # Totals
    st.subheader("💰 Totals")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Subtotal", f"{data.get('currency', '')} {data.get('subtotal', 'N/A')}")
    with col2:
        st.metric("Tax Amount", f"{data.get('currency', '')} {data.get('tax_amount', 'N/A')}")
    with col3:
        st.metric("Total Amount", f"{data.get('currency', '')} {data.get('total_amount', 'N/A')}")

# Main app
st.title("📄 Invoice Data Extractor")
st.markdown("Upload an invoice image and extract structured data automatically!")

# Sidebar for API key
with st.sidebar:
    st.header("🔑 Setup")
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
    
    if api_key:
        st.secrets["OPENAI_API_KEY"] = api_key
        st.success("API key configured!")
    
    st.header("📊 Instructions")
    st.markdown("""
    1. Enter your OpenAI API key
    2. Upload an invoice image
    3. Click 'Extract Invoice Data'
    4. View and download results
    """)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an invoice image...", 
    type=['png', 'jpg', 'jpeg'],
    help="Upload PNG, JPG, or JPEG files"
)

if uploaded_file is not None:
    # Display the uploaded image
    st.subheader("📸 Uploaded Invoice")
    
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Invoice", use_column_width=True)
        
        # Check if API key is provided
        if not st.secrets.get("OPENAI_API_KEY"):
            st.warning("Please enter your OpenAI API key in the sidebar to proceed.")
        else:
            # Extract data button
            if st.button("🔍 Extract Invoice Data", type="primary"):
                with st.spinner("Analyzing invoice..."):
                    extracted_data = extract_invoice_data(image)
                    
                    if extracted_data:
                        st.success("Data extracted successfully!")
                        
                        # Display extracted data
                        display_extracted_data(extracted_data)
                        
                        # Download options
                        st.subheader("📥 Download Options")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # JSON download
                            json_str = json.dumps(extracted_data, indent=2)
                            st.download_button(
                                label="📄 Download JSON",
                                data=json_str,
                                file_name="invoice_data.json",
                                mime="application/json"
                            )
                        
                        with col2:
                            # CSV download (line items)
                            if extracted_data.get("line_items"):
                                df = pd.DataFrame(extracted_data["line_items"])
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    label="📊 Download Line Items CSV",
                                    data=csv,
                                    file_name="invoice_line_items.csv",
                                    mime="text/csv"
                                )
                        
                        # Raw JSON data (expandable)
                        with st.expander("🔍 View Raw JSON Data"):
                            st.json(extracted_data)
                        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and OpenAI Vision API")
