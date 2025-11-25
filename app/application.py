from flask import Flask, render_template, request, session, redirect, url_for
from app.components.retriever import create_qa_chain
from app.common.logger import get_logger
from dotenv import load_dotenv
import os

logger = get_logger(__name__)

load_dotenv()

# Get the directory where this file is located
basedir = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(basedir, 'templates')

app = Flask(__name__, template_folder=template_dir)
app.secret_key = os.getenv('SECRET_KEY', 'medical-rag-chatbot-secret-key-2024')  # Fixed key
# Session expires when browser is closed (not permanent)
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_TYPE'] = 'filesystem'

from markupsafe import Markup

def nl2br(value):
    return Markup(value.replace("\n", "<br>\n"))

app.jinja_env.filters['nl2br'] = nl2br

@app.route('/', methods=['GET', 'POST'])
def index():
    # Session is NOT permanent - expires when browser closes
    if "messages" not in session: #This can be user as well as LLM messages
        session['messages'] = []
    
    if request.method == 'POST':
        user_input = request.form.get('prompt')
        logger.info(f"Received POST request with input: {user_input}")
        if user_input:
            messages = session.get('messages', [])
            messages.append({"role": "user", "content": user_input})
            session['messages'] = messages
            session.modified = True

            try:
                qa_chain = create_qa_chain(chat_history=messages)
                response = qa_chain.invoke({"question": user_input})
                result = response.get("answer", "No response")
                messages.append({"role": "assistant", "content": result})
                session['messages'] = messages
                session.modified = True

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                return render_template('index.html', messages = session['messages'], error = error_msg)
        
        return redirect(url_for('index'))
    
    return render_template('index.html', messages = session.get('messages', []))


@app.route('/clear')
def clear():
    session.pop('messages', None)
    return redirect(url_for('index')) #it is function index and not index.html


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000, debug = False, use_reloader = False)
