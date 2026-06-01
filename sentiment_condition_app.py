import os
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="SentimentFlow",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');
:root {
    --bg:#080810;--surface:#0f0f1a;--card:#13131f;--border:#1f1f30;
    --p:#7b6ef6;--pl:#a78bfa;--pos:#34d399;--neg:#f87171;
    --text:#e4e4f0;--muted:#5a5a72;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text);font-family:'JetBrains Mono',monospace;}
[data-testid="stHeader"],[data-testid="stToolbar"]{display:none!important;}
#MainMenu,footer{display:none;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
textarea{background:var(--card)!important;border:1px solid var(--border)!important;color:var(--text)!important;font-family:'JetBrains Mono',monospace!important;font-size:.88rem!important;border-radius:10px!important;}
textarea:focus{border-color:var(--p)!important;box-shadow:0 0 0 3px #7b6ef625!important;}
[data-testid="stButton"]>button{background:linear-gradient(135deg,var(--p),var(--pl))!important;color:#fff!important;border:none!important;border-radius:10px!important;font-family:'Syne',sans-serif!important;font-weight:700!important;letter-spacing:.04em;padding:.55rem 1.4rem!important;transition:opacity .2s,transform .15s;}
[data-testid="stButton"]>button:hover{opacity:.85;transform:translateY(-1px);}
[data-testid="stProgress"]>div>div{background:var(--p)!important;}
p,label,span,li,h1,h2,h3{color:var(--text)!important;}
.stMarkdown p{color:var(--text)!important;}
</style>
""", unsafe_allow_html=True)

for k,v in [("history",[]),("result",None),("running",False),("pipeline_log",[]),("stats",{"total":0,"positive":0,"negative":0})]:
    if k not in st.session_state: st.session_state[k]=v

@st.cache_resource(show_spinner=False)
def build_workflow():
    from langchain_openai import AzureChatOpenAI
    from pydantic import BaseModel,Field
    from typing import TypedDict,Literal
    from langgraph.graph import StateGraph,START,END

    model=AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("api_version"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_MODEL"),
    )

    class Str_Model(BaseModel):
        sentiment:Literal["Positive","Negative"]=Field(description="Give Sentiment Of The Review")
        domain:Literal["Software","Business","Education","Social Media"]=Field(description="By Analysing the Review, Say Which Domain It Is")

    str_model=model.with_structured_output(Str_Model)

    class Sentiment(TypedDict):
        topic:str; sentiment:str; domain:str; msg:str

    def sentiment_node(state):
        result=str_model.invoke(f"Based On the Topic '{state['topic']}' Decide Its Sentiment")
        return {"sentiment":result.sentiment,"domain":result.domain}

    def route(state): return state["sentiment"]

    def positive_node(state):
        return {"msg":model.invoke(f"Based On the topic '{state['topic']}' with Sentiment Positive and Domain {state['domain']} give Me a happiest msg").content}

    def negative_node(state):
        return {"msg":model.invoke(f"Based On the topic '{state['topic']}' with Sentiment Negative and Domain {state['domain']} give Me a sad msg").content}

    graph=StateGraph(Sentiment)
    graph.add_node("sentiment",sentiment_node)
    graph.add_node("Positive",positive_node)
    graph.add_node("Negative",negative_node)
    graph.add_edge(START,"sentiment")
    graph.add_conditional_edges("sentiment",route,{"Positive":"Positive","Negative":"Negative"})
    graph.add_edge("Positive",END)
    graph.add_edge("Negative",END)
    return graph.compile()

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 .5rem'>
      <p style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;
         background:linear-gradient(135deg,#7b6ef6,#f6c96e);
         -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0'>SentimentFlow</p>
      <p style='font-size:.7rem;color:#5a5a72;letter-spacing:.1em;text-transform:uppercase;margin:2px 0 0'>LangGraph · Azure OpenAI</p>
    </div>""", unsafe_allow_html=True)
    st.divider()

    s=st.session_state.stats
    total=s["total"]
    pos_pct=round((s["positive"]/total*100) if total else 0)
    st.markdown("**Session stats**")
    c1,c2=st.columns(2)
    c1.metric("Total",total)
    c2.metric("Positive",f"{pos_pct}%")
    if total: st.progress(pos_pct/100)
    st.divider()

    st.markdown("**Quick examples**")
    examples={
        "🚀 Software":"The new VS Code update broke all my extensions",
        "📈 Business":"Our startup just closed a $10M Series A round!",
        "📚 Education":"Online learning has transformed remote education beautifully",
        "📱 Social":"Twitter's algorithm keeps showing me irrelevant ads",
    }
    for label,text in examples.items():
        if st.button(label,use_container_width=True):
            st.session_state["prefill"]=text
            st.rerun()
    st.divider()

    if st.session_state.history:
        st.markdown("**History**")
        for h in st.session_state.history[:6]:
            color="#34d399" if h["sentiment"]=="Positive" else "#f87171"
            icon="😄" if h["sentiment"]=="Positive" else "😔"
            st.markdown(
                f"<div style='border-left:2px solid {color};padding:6px 10px;margin-bottom:6px;"
                f"border-radius:0 6px 6px 0;background:#13131f;font-size:.72rem'>"
                f"<span style='color:{color}'>{icon} {h['sentiment']}</span> · "
                f"<span style='color:#a5b4fc'>{h['domain']}</span><br>"
                f"<span style='color:#5a5a72'>{h['topic'][:40]}…</span></div>",
                unsafe_allow_html=True)
        if st.button("🗑 Clear history",use_container_width=True):
            st.session_state.history=[]
            st.session_state.stats={"total":0,"positive":0,"negative":0}
            st.rerun()

# ── MAIN ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:2.5rem 0 1rem'>
  <p style='font-family:Syne,sans-serif;font-size:clamp(2rem,5vw,3.5rem);font-weight:800;
     letter-spacing:-.03em;background:linear-gradient(135deg,#7b6ef6 0%,#a78bfa 40%,#f6c96e 100%);
     -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
     margin:0;line-height:1.1'>Sentiment Analysis</p>
  <p style='font-size:.8rem;color:#5a5a72;letter-spacing:.1em;text-transform:uppercase;margin-top:6px'>
     powered by LangGraph + Azure OpenAI</p>
</div>""", unsafe_allow_html=True)

prefill=st.session_state.pop("prefill","")
left,right=st.columns([2,1],gap="large")

with left:
    st.markdown("#### Your review or topic")
    review=st.text_area("review_input",value=prefill,
        placeholder="Type a product review, news headline, or any topic…",
        height=140,label_visibility="collapsed")
    c1,c2,c3=st.columns([2,1,1])
    with c1: analyse=st.button("⚡  Analyse",use_container_width=True)
    with c2:
        if st.button("✕  Clear",use_container_width=True):
            st.session_state.result=None
            st.session_state.pipeline_log=[]
            st.rerun()
    with c3: compare_mode=st.checkbox("Compare mode",value=False)

with right:
    st.markdown("#### Pipeline map")
    nodes=[("START","#5a5a72"),("sentiment","#7b6ef6"),("Positive / Negative","#34d399"),("END","#5a5a72")]
    for name,color in nodes:
        active=""
        if st.session_state.result:
            sent=st.session_state.result.get("sentiment","")
            if name=="sentiment" or (name=="Positive / Negative"): active=f"background:{color}18;"
        st.markdown(
            f"<div style='border:1px solid {color}40;border-left:3px solid {color};"
            f"border-radius:6px;padding:7px 12px;margin-bottom:7px;"
            f"font-size:.78rem;color:{color};{active}'>{name}</div>",
            unsafe_allow_html=True)

if compare_mode:
    st.markdown("---")
    st.markdown("#### Compare two topics")
    cc1,cc2=st.columns(2)
    with cc1: review_a=st.text_area("Topic A",placeholder="First topic…",height=80,key="cmp_a")
    with cc2: review_b=st.text_area("Topic B",placeholder="Second topic…",height=80,key="cmp_b")
    compare_btn=st.button("⚡  Compare both",use_container_width=True)
else:
    compare_btn=False

st.markdown("---")

def show_pipeline_log(log):
    if not log: return
    st.markdown("**Pipeline trace**")
    for entry in log:
        color="#34d399" if entry["status"]=="done" else "#7b6ef6"
        st.markdown(
            f"<div style='font-size:.78rem;color:{color};padding:3px 0;font-family:JetBrains Mono,monospace'>"
            f"✓ [{entry['ts']}]  {entry['node']}</div>",
            unsafe_allow_html=True)

def run_analysis(topic):
    log=[]
    def add_log(node): log.append({"ts":time.strftime("%H:%M:%S"),"node":node,"status":"done"})
    try: workflow=build_workflow()
    except Exception as e: st.error(f"Model init failed: {e}"); return None,log

    steps=[
        ("Invoking sentiment node…","sentiment → classifying topic"),
        ("Routing on sentiment…","conditional_edge → Positive/Negative"),
        ("Generating response…","response node → crafting message"),
    ]
    prog=st.progress(0,text="Starting pipeline…")
    for i,(msg,node_label) in enumerate(steps):
        prog.progress((i+1)/(len(steps)+1),text=msg)
        add_log(node_label)
        time.sleep(0.25)
    try:
        final=workflow.invoke({"topic":topic})
        prog.progress(1.0,text="Pipeline complete ✓")
        time.sleep(0.35)
        prog.empty()
        return final,log
    except Exception as e:
        prog.empty()
        st.error(f"Pipeline error: {e}")
        return None,log

def render_result(result,label=""):
    if not result: return
    s=result.get("sentiment",""); d=result.get("domain",""); msg=result.get("msg","")
    clr="#34d399" if s=="Positive" else "#f87171"
    icon="😄" if s=="Positive" else "😔"
    header=f"<b>{label}</b><br>" if label else ""
    st.markdown(
        f"""<div style='background:#13131f;border:1px solid #1f1f30;border-radius:12px;padding:1.4rem 1.6rem;margin-bottom:1rem'>
        <div style='font-size:.7rem;color:#5a5a72;text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px'>{header}Analysis result</div>
        <div style='display:flex;gap:12px;margin-bottom:14px;flex-wrap:wrap'>
          <div style='background:#0a0a10;border:1px solid #1f1f30;border-radius:8px;padding:10px 16px;text-align:center;flex:1;min-width:100px'>
            <div style='font-size:.65rem;color:#5a5a72;text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px'>Sentiment</div>
            <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:{clr}'>{icon} {s}</div>
          </div>
          <div style='background:#0a0a10;border:1px solid #1f1f30;border-radius:8px;padding:10px 16px;text-align:center;flex:1;min-width:100px'>
            <div style='font-size:.65rem;color:#5a5a72;text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px'>Domain</div>
            <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:#a5b4fc'>🏷 {d}</div>
          </div>
        </div>
        <hr style='border:none;border-top:1px solid #1f1f30;margin:12px 0'>
        <p style='font-size:.9rem;line-height:1.8;color:#e4e4f0;white-space:pre-wrap;margin:0'>{msg}</p>
        </div>""",
        unsafe_allow_html=True)

# ── Single analysis ────────────────────────────────────────────────────────────
if analyse and not compare_mode:
    if not review.strip(): st.warning("Please enter a review or topic.")
    else:
        with st.spinner(""):
            result,log=run_analysis(review.strip())
        if result:
            st.session_state.result=result
            st.session_state.pipeline_log=log
            sent=result["sentiment"]
            st.session_state.stats["total"]+=1
            st.session_state.stats[sent.lower()]+=1
            st.session_state.history.insert(0,{"topic":review.strip()[:60],"sentiment":sent,"domain":result["domain"],"msg":result["msg"][:120]})
            if len(st.session_state.history)>20: st.session_state.history=st.session_state.history[:20]

# ── Compare analysis ───────────────────────────────────────────────────────────
if compare_btn:
    for topic,lbl in [(review_a.strip(),"Topic A"),(review_b.strip(),"Topic B")]:
        if not topic: st.warning(f"{lbl} is empty — skipping."); continue
        st.markdown(f"**Running {lbl}…**")
        result,log=run_analysis(topic)
        if result:
            render_result(result,lbl)
            sent=result["sentiment"]
            st.session_state.stats["total"]+=1
            st.session_state.stats[sent.lower()]+=1
            st.session_state.history.insert(0,{"topic":topic[:60],"sentiment":sent,"domain":result["domain"],"msg":result["msg"][:120]})

# ── Show result + log ──────────────────────────────────────────────────────────
if st.session_state.result and not compare_mode:
    render_result(st.session_state.result)
    show_pipeline_log(st.session_state.pipeline_log)

# ── Distribution chart ─────────────────────────────────────────────────────────
if st.session_state.stats["total"]>=2:
    import streamlit.components.v1 as components
    s=st.session_state.stats
    pos=s["positive"]; neg=s["negative"]; total=s["total"]
    pos_w=round(pos/total*100); neg_w=100-pos_w
    st.markdown("---")
    st.markdown("#### Session distribution")
    components.html(f"""
    <div style="font-family:'JetBrains Mono',monospace;color:#e4e4f0;padding:8px 0">
      <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px">
        <span style="font-size:.75rem;color:#5a5a72;width:80px">Positive</span>
        <div style="flex:1;height:20px;background:#1f1f30;border-radius:4px;overflow:hidden">
          <div style="width:{pos_w}%;height:100%;background:#34d399;border-radius:4px"></div>
        </div>
        <span style="font-size:.8rem;color:#34d399;width:40px;text-align:right">{pos}</span>
      </div>
      <div style="display:flex;gap:8px;align-items:center">
        <span style="font-size:.75rem;color:#5a5a72;width:80px">Negative</span>
        <div style="flex:1;height:20px;background:#1f1f30;border-radius:4px;overflow:hidden">
          <div style="width:{neg_w}%;height:100%;background:#f87171;border-radius:4px"></div>
        </div>
        <span style="font-size:.8rem;color:#f87171;width:40px;text-align:right">{neg}</span>
      </div>
    </div>""", height=80)