import streamlit as st
from query import ask_question

st.title("📄 Chat with PDF")

query = st.text_input("Ask your question:")

if st.button("Search"):
    st.write("🔍 Processing...")

    try:
        result = ask_question(query)
           # 👈 IMPORTANT

        if result and "answer" in result:
            st.write("### Answer:")
            st.write(result["answer"])
        else:
            st.write("❌ No answer returned")

    except Exception as e:
        st.error(f"ERROR: {str(e)}")