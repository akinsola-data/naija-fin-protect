import os
import streamlit as st
import joblib

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="NaijaFinProtect — Multilingual Fraud Detection",
    layout="wide"
)

# Minimalist Corporate Typography and Styling (Compatible with Dark & Light Mode)
st.markdown("""
<style>
    .main-title { font-size: 2.4rem; font-weight: 700; margin-bottom: 5px; }
    .sub-title { font-size: 1.1rem; font-weight: 400; color: #6B7280; margin-bottom: 25px; }
    .header-text { font-size: 1.3rem; font-weight: 600; margin-top: 15px; margin-bottom: 10px; }
    .section-divider { margin-top: 20px; margin-bottom: 20px; border-top: 1px solid #374151; }
</style>
""", unsafe_allow_html=True)

# Main Headers
st.markdown('<div class="main-title">NaijaFinProtect: Multilingual Fraud Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enterprise NLP microservice for detecting phishing, loan shark harassment, and financial fraud across African languages (Pidgin, Yoruba, Igbo, Hausa, English).</div>', unsafe_allow_html=True)

# Load Production Model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "../models/afri_fraud_model.joblib")
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model pipeline: {e}. Ensure model training script has been executed.")
        return None

model = load_model()

# Sidebar - Preset Samples for Testing
st.sidebar.markdown('<div class="header-text">Select Sample Text</div>', unsafe_allow_html=True)
st.sidebar.markdown("Choose a real-world Nigerian banking communication sample below:")

samples = {
    "Select a sample...": "",
    "Pidgin - Phishing Alert": "Omo your Opay account don enter hold balance because of BVN linkage. Make you click dis link http://opay-bvn-ng.xyz right now make dem no chop your money.",
    "Pidgin - Genuine Transaction": "Bros the alert don drop! 50k clear money inside my First Bank account. God bless you plenty bro.",
    "Yoruba - Impersonation Scam": "Iro ni won pa! Iwe iroyin lati EFCC wipe a ti block account re ni GTBank. Pe nọmba yii 08031234567 lati ṣii bayi ki owo re ma baa sọnu.",
    "Yoruba - Genuine Inflow": "A ti sanwo ile iwe fun omo re. E dupe lowo Olorun. Oja tita ti wọle sinu akọọlẹ mi ni Access Bank.",
    "Igbo - Loan Harassment": "I zuru ohi! I gbaziri ego na SpeedCash weghachighị ya. Ọ bụrụ na ị kwụghị ụgwọ tupu elekere atọ, anyị ga-egosi ndị ezi-na-ụlọ gị na ị bụ onye ohi.",
    "Igbo - Genuine Transfer": "Alereti ebatala! Ego NGN 40,000 agafeela n'akaunti mi nke UBA. Chukwu gozie gị nwanna.",
    "Hausa - Fake Lottery Scam": "Barka da asuba! Lambar wayarka ta ci NGN 150,000 a shirin agaji na musamman. Aiko da BVN dinka da lambar asusu don karbar kudinka.",
    "Hausa - Genuine Deposit": "An tura kudi NGN 50,000 zuwa asusunka na Jaiz Bank cikin nasara. Nagode da kasuwanci, Allah ya kara budi.",
    "English - BVN Verification Scam": "Dear Customer, your GTBank account has been restricted due to incomplete BVN update. Click http://gtb-update-ng.com to verify immediately."
}

selected_sample = st.sidebar.selectbox("Preset Samples", list(samples.keys()), label_visibility="collapsed")
sample_text = samples[selected_sample]

# Main Input Section
st.markdown('<div class="header-text">Message Verification</div>', unsafe_allow_html=True)
user_input = st.text_area("Paste the text message or banking alert below for evaluation:", value=sample_text, height=140)

if st.button("Analyze Text", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a text message to initiate verification.")
    elif not model:
        st.error("Model pipeline is not loaded on the server.")
    else:
        with st.spinner("Processing text..."):
            # Execute Model Prediction
            probabilities = model.predict_proba([user_input])[0]
            fraud_prob = float(probabilities[1])
            is_fraud = fraud_prob >= 0.5
            
            # Result Section
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="header-text">Risk Assessment Report</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if fraud_prob >= 0.75:
                    st.error(f"**HIGH RISK DETECTED**\n\n**Category:** Phishing / Financial Fraud / Extortion\n\n**Action Plan:** Do not click any links, do not provide verification details (OTP/PIN), and restrict communications with the sender immediately.")
                elif fraud_prob >= 0.5:
                    st.warning(f"**SUSPICIOUS ACTIVITY DETECTED**\n\n**Category:** Potential Phishing or Impersonation\n\n**Action Plan:** The message exhibits common phishing structural patterns. Verify independently through official banking channels.")
                else:
                    st.success(f"**NO THREAT DETECTED**\n\n**Category:** Genuine Transaction / Customer Support\n\n**Action Plan:** The message conforms to standard financial communication syntax with no indicators of malicious intent.")
            
            with col2:
                # Using native Streamlit metrics for a clean corporate appearance
                st.metric(label="Fraud Confidence Score", value=f"{fraud_prob*100:.1f}%")
                st.metric(label="Genuine Confidence Score", value=f"{(1-fraud_prob)*100:.1f}%")
                
            # Details expander with professional copy
            with st.expander("Inspection Details & Pipeline Metrics"):
                st.markdown(f"- **Message Length:** {len(user_input)} characters")
                st.markdown(f"- **Feature Extraction:** Bigram TF-IDF Sublinear Vectorization")
                st.markdown(f"- **Classification Engine:** Logistic Regression (Optimized Class Weights)")
                st.markdown(f"- **Calculated Risk Index:** {fraud_prob:.4f}")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6B7280; font-size: 0.85rem;'>Developed by Akinsola Emmanuel • Built with Streamlit & Scikit-Learn • Deployed via Hugging Face Spaces</p>", unsafe_allow_html=True)