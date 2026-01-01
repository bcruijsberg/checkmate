import streamlit as st
import pandas as pd

# Set page configuration
st.set_page_config(layout="wide", page_title="Model Output Comparison")

@st.cache_data
def load_data():
    df = pd.read_csv("eval_gptoss_output.csv")
    return df

def main():
       
    df = load_data()
    
    # 1. Identify Available Models
    # Look for suffixes in columns starting with 'checkable_'
    model_suffixes = [col.replace("checkable_", "") for col in df.columns if col.startswith("checkable_")]
    
    # 2. Sidebar Configuration
    st.sidebar.header("Settings")
    selected_model = st.sidebar.selectbox("Select Model to Compare", model_suffixes)
    
    # Navigation for rows
    row_index = st.sidebar.number_input("Select Row Index", min_value=0, max_value=len(df)-1, value=0)
    
    row = df.iloc[row_index]
    
    st.title(row['claim'])

    # 4. Comparison Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Baseline (Ground Truth)")
        st.write(f"**Checkable:** `{row['checkable']}`")
        st.write("**Explanation:**")
        st.write(row['explanation'])
        st.write("**Details Text:**")
        st.text(row['details_text'])
        st.write("**Alerts:**")
        st.warning(row['alerts'] if pd.notna(row['alerts']) else "No alerts")

    with col2:
        st.subheader(f"🤖 Model: {selected_model}")
        
        # Dynamic column names based on selection
        m_checkable = f"checkable_{selected_model}"
        m_explanation = f"explanation_{selected_model}"
        m_details = f"details_{selected_model}"
        m_alerts = f"alerts_{selected_model}"
        
        st.write(f"**Checkable:** `{row.get(m_checkable, 'N/A')}`")
        st.write("**Explanation:**")
        st.write(row.get(m_explanation, 'N/A'))
        st.write("**Details Text:**")
        st.text(row.get(m_details, 'N/A'))
        st.write("**Alerts:**")
        st.warning(row.get(m_alerts, 'No alerts') if pd.notna(row.get(m_alerts)) else "No alerts")

    st.divider()

    # 5. Model Metrics / Validation Flags
    st.subheader(f"✅ Validation Flags for {selected_model}")
    
    # Mapping the check columns
    metrics = {
        "Reasoning": f"reason_{selected_model}",
        "Completeness": f"complete_{selected_model}",
        "Hallucination Free": f"halluci_{selected_model}",
        "Intent Correct": f"intent_{selected_model}"
    }
    
    m_cols = st.columns(len(metrics))
    for i, (label, col_name) in enumerate(metrics.items()):
        val = row.get(col_name)
        with m_cols[i]:
            if val is True or str(val).lower() == 'true':
                st.success(f"{label}: ✅")
            elif val is False or str(val).lower() == 'false':
                st.error(f"{label}: ❌")
            else:
                st.info(f"{label}: {val}")

    # 6. Interaction Context
    with st.expander("Show User Interaction Context"):
        st.write(f"**Question Asked:** {row['question']}")
        st.write(f"**User Answer:** {row['user_answer']}")
        st.write(f"**Confirmed Move On:** {'Yes' if row['confirmed'] else 'No'}")

if __name__ == "__main__":
    main()