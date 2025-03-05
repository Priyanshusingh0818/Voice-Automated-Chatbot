from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch
import random
import json
import re
from datetime import datetime

app = Flask(__name__)

# Initialize models with explicit PyTorch backend
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available

# Initialize multiple models for different types of questions
qa_model = pipeline('question-answering', 
                   model='deepset/roberta-base-squad2',
                   device=device,
                   framework='pt')  # Explicitly specify PyTorch

# For the general model, you can either use GPT-2 with PyTorch
general_model = pipeline('text-generation', 
                        model='gpt2',
                        device=device,
                        framework='pt')

# Rest of your code remains the same...

# Comprehensive knowledge base
KNOWLEDGE_BASE = {
    "general_ai": """
    Artificial Intelligence (AI) is the simulation of human intelligence by machines. It encompasses various subfields:
    - Machine Learning: Systems that learn and improve from experience
    - Deep Learning: Neural network-based learning systems
    - Natural Language Processing (NLP): Understanding and generating human language
    - Computer Vision: Processing and analyzing visual information
    - Robotics: Machines that can interact with the physical world
    - Expert Systems: Programs that emulate decision-making of human experts
    """,
    
    "technical": """
    Technical concepts in AI include:
    - Neural Networks: Computing systems inspired by biological neural networks
    - Algorithms: Step-by-step procedures for solving problems
    - Data Structures: Organizations of data for efficient access and modification
    - Programming Languages: Tools for creating software
    - Cloud Computing: Remote computing resources
    - Cybersecurity: Protection of systems and data
    """,
    
    "applications": """
    AI applications in various fields:
    - Healthcare: Disease diagnosis, drug discovery, patient care
    - Finance: Fraud detection, trading algorithms, risk assessment
    - Education: Personalized learning, automated grading
    - Transportation: Self-driving vehicles, traffic management
    - Entertainment: Game AI, content recommendation
    - Business: Customer service, process automation, analytics
    """,
    
    "ethics": """
    AI Ethics and considerations:
    - Privacy: Protection of personal data
    - Bias: Ensuring fair and unbiased AI systems
    - Transparency: Understanding AI decision-making
    - Accountability: Responsibility for AI actions
    - Safety: Ensuring AI systems are safe and reliable
    - Social Impact: Effects on society and employment
    """
}

# Personality traits for more engaging responses
PERSONALITY_TRAITS = {
    "friendly": [
        "I'm happy to help you understand this better!",
        "That's a great question!",
        "Let me explain this in a clear way.",
        "I'm excited to share what I know about this.",
    ],
    "professional": [
        "Based on my analysis,",
        "Here's a comprehensive explanation:",
        "Let me break this down for you:",
        "From a technical perspective,",
    ],
    "educational": [
        "An interesting aspect of this is...",
        "To put this in context,",
        "Here's a helpful way to think about it:",
        "Let me illustrate this with an example:",
    ]
}

def clean_text(text):
    """Clean and format text for better readability"""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def get_personality_response():
    """Get a random personality response"""
    category = random.choice(list(PERSONALITY_TRAITS.keys()))
    return random.choice(PERSONALITY_TRAITS[category])

def get_relevant_context(question):
    """Get the most relevant context based on the question"""
    question = question.lower()
    context_parts = []
    
    # Keywords for different categories
    categories = {
        "general_ai": ["ai", "artificial intelligence", "machine learning", "what is", "define"],
        "technical": ["how", "technical", "neural", "algorithm", "code", "program"],
        "applications": ["use", "application", "industry", "real world", "example"],
        "ethics": ["ethics", "privacy", "bias", "fair", "safety", "social"]
    }
    
    # Add relevant context based on keywords
    for category, keywords in categories.items():
        if any(keyword in question for keyword in keywords):
            context_parts.append(KNOWLEDGE_BASE[category])
    
    # If no specific category matches, use all context
    if not context_parts:
        context_parts = list(KNOWLEDGE_BASE.values())
    
    return " ".join(context_parts)

def generate_enhanced_response(question, answer, confidence):
    """Generate a more comprehensive and engaging response"""
    # Start with a personality-driven opener
    response_parts = [get_personality_response()]
    
    # Add the main answer
    response_parts.append(answer)
    
    # Add relevant examples or additional context
    if confidence > 0.8:
        response_parts.append("\n\nThis is a well-established concept in the field.")
    elif confidence > 0.5:
        response_parts.append("\n\nWhile this is my current understanding, I encourage you to explore this topic further.")
    else:
        response_parts.append("\n\nThis is a complex topic with many perspectives. I've provided my best understanding.")
    
    # Add followup suggestions
    if "what" in question.lower():
        response_parts.append("\n\nYou might also be interested in learning about how this is applied in practice.")
    elif "how" in question.lower():
        response_parts.append("\n\nIf you'd like, we can explore specific examples or use cases.")
    
    return " ".join(response_parts)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Get relevant context
        context = get_relevant_context(question)
        
        # Get initial answer using the QA model
        qa_result = qa_model(question=question, context=context)
        
        # Generate enhanced response
        enhanced_answer = generate_enhanced_response(
            question,
            qa_result['answer'],
            qa_result['score']
        )
        
        # Log the interaction (optional)
        print(f"[{datetime.now()}] Q: {question} | Confidence: {qa_result['score']}")
        
        return jsonify({
            'answer': enhanced_answer,
            'confidence': round(float(qa_result['score']) * 100, 2)
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'error': 'I apologize, but I encountered an error processing your question. '
                    'Could you please rephrase it or ask something else?'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)