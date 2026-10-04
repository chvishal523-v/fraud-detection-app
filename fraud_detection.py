import joblib
import pandas as pd
import streamlit as st
from google import genai

# ---------- Page setup ----------
st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="wide")

GEMINI_MODEL = "gemini-2.5-flash"  # change here if you want another Gemini model


@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_pipeline.pkl")


@st.cache_resource
def load_gemini():
    key = st.secrets.get("GEMINI_API_KEY")
    return genai.Client(api_key=key) if key else None


model = load_model()
gemini = load_gemini()


# ---------- Simple rule-based red flags (the "facts" we hand to Gemini) ----------
def find_red_flags(t, amount, old_org, new_org, old_dest, new_dest):
    flags = []
    if t in ("TRANSFER", "CASH_OUT"):
        flags.append(f"Transaction type is {t} (the only types where fraud usually happens).")
    if old_org > 0 and new_org == 0:
        flags.append("Sender's account was completely emptied.")
    if abs((old_org - amount) - new_org) > 1:
        flags.append("Sender's balance change does not match the amount sent.")
    if old_dest == 0 and new_dest == 0 and amount > 0:
        flags.append("Receiver account had zero balance before and after (looks like a mule/empty account).")
    if abs((old_dest + amount) - new_dest) > 1 and new_dest > 0:
        flags.append("Receiver's balance change does not match the amount received.")
    if amount > old_org:
        flags.append("Amount is bigger than the sender's balance.")
    return flags


def explain_with_gemini(inputs, label, prob, flags):
    if gemini is None:
        return "Gemini API key not set. Add GEMINI_API_KEY in Streamlit secrets."
    prompt = f"""You are a bank fraud analyst explaining a model result to a non-technical person.
Use simple words and one everyday analogy.

Transaction: {inputs}
Model verdict: {label} (fraud probability {prob:.1%})
Red flags found by rules: {flags if flags else 'none'}

Write:
1. What kind of fraud this most resembles (account takeover, mule account, cash-out after transfer, etc.) or why it looks normal.
2. Why the model likely said this, based only on the data above.
3. 3 clear next steps for the bank or customer.
Keep it under 150 words. Do not invent facts that are not in the data."""
    try:
        resp = gemini.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return resp.text
    except Exception as e:
        return f"Could not get an explanation right now: {e}"


# ---------- Header ----------
st.title("🛡️ Bank Fraud Detection")
st.caption("Enter a transaction, get a fraud prediction, and an AI explanation in plain language.")
st.divider()

# ---------- Input form ----------
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Transaction details")
    transaction_type = st.selectbox("Transaction type", ["PAYMENT", "TRANSFER", "CASH_OUT"])
    amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=100.0)

with right:
    st.subheader("Account balances")
    c1, c2 = st.columns(2)
    with c1:
        oldbalanceOrg = st.number_input("Sender - old balance", min_value=0.0, value=10000.0)
        newbalanceOrig = st.number_input("Sender - new balance", min_value=0.0, value=9000.0)
    with c2:
        oldbalanceDest = st.number_input("Receiver - old balance", min_value=0.0, value=0.0)
        newbalanceDest = st.number_input("Receiver - new balance", min_value=0.0, value=0.0)

predict = st.button("🔍 Check transaction", type="primary", use_container_width=True)

# ---------- Prediction ----------
if predict:
    row = {
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
    }
    input_df = pd.DataFrame([row])

    pred = int(model.predict(input_df)[0])
    prob = float(model.predict_proba(input_df)[0][1]) if hasattr(model, "predict_proba") else float(pred)
    label = "FRAUD" if pred == 1 else "NOT FRAUD"
    flags = find_red_flags(transaction_type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Verdict", label)
    m2.metric("Fraud probability", f"{prob:.1%}")
    m3.metric("Red flags found", len(flags))
    st.progress(min(max(prob, 0.0), 1.0))

    if pred == 1:
        st.error("⚠️ This transaction looks like fraud.")
    else:
        st.success("✅ This transaction looks safe.")

    tab1, tab2 = st.tabs(["🤖 AI explanation", "🚩 Red flags"])
    with tab1:
        with st.spinner("Gemini is analysing..."):
            st.write(explain_with_gemini(row, label, prob, flags))
    with tab2:
        if flags:
            for f in flags:
                st.warning(f)
        else:
            st.info("No obvious red flags in the numbers.")
