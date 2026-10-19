
import string
from flask import Flask, render_template, request, jsonify
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

faqs = [
    {
        "question": "How can I track my order?",
        "answer": "You can track your order using the tracking link sent to your registered email."
    },
    {
        "question": "What is your return policy?",
        "answer": "You can return eligible products within 7 days of delivery."
    },
    {
        "question": "How long does delivery take?",
        "answer": "Delivery usually takes 3 to 5 business days."
    },
    {
        "question": "Which payment methods do you accept?",
        "answer": "We accept UPI, debit cards, credit cards, and net banking."
    },
    {
        "question": "How can I cancel my order?",
        "answer": "You can cancel your order before it is shipped by visiting your orders page."
    },
    {
        "question": "How can I contact customer support?",
        "answer": "You can contact customer support through the Help section on our website."
    },
    {
        "question": "Can I get a refund?",
        "answer": "Refunds are processed after the returned product is approved, according to the refund policy."
    },
    {
        "question": "Do you offer free shipping?",
        "answer": "Free shipping is available on eligible orders. Check the shipping details at checkout."
    },
    {
        "question": "Can I change my delivery address?",
        "answer": "You may be able to change your delivery address before your order is shipped."
    },
    {
        "question": "How do I create an account?",
        "answer": "Click the Sign Up button on our website and enter the required details."
    },
    
    {
        "question": "Do you provide order confirmation?",
        "answer": "Yes, an order confirmation is sent to your registered email after a successful purchase."
    },
    {
        "question": "What should I do if my order is delayed?",
        "answer": "Please check your tracking details for the latest update. Contact customer support if the delay continues."
    },
    {
        "question": "Can I track my order without an account?",
        "answer": "You may be able to track your order using the tracking link sent to your registered email."
    },
    {
        "question": "Can I change my order after placing it?",
        "answer": "Order changes may be possible before processing or shipment. Please contact customer support promptly."
    },
    {
        "question": "What happens if I receive a damaged product?",
        "answer": "Please contact customer support with your order details and photos of the damaged product."
    },
    {
        "question": "What if I receive the wrong item?",
        "answer": "Contact customer support with your order details so the issue can be reviewed."
    },
    {
        "question": "How long does a refund take?",
        "answer": "Refund processing time depends on the payment method and the store's refund policy."
    },
    {
        "question": "Can I exchange a product?",
        "answer": "Product exchanges depend on eligibility and availability. Please check the store's exchange policy."
    },
    {
        "question": "Do you offer cash on delivery?",
        "answer": "Cash on delivery may be available for eligible orders and delivery locations."
    },
    {
        "question": "Is online payment safe?",
        "answer": "Use the store's official checkout and trusted payment methods to complete your purchase securely."
    },
    {
        "question": "Can I use multiple discount coupons?",
        "answer": "Coupon combination rules depend on the offer terms. Check the conditions before checkout."
    },
    {
        "question": "How can I apply a coupon code?",
        "answer": "Enter your coupon code in the discount or promo code field during checkout."
    },
    {
        "question": "Do you offer discounts?",
        "answer": "Available discounts and promotions are displayed on the store's website."
    },
    {
        "question": "How can I update my account details?",
        "answer": "Sign in to your account and open the profile or account settings section."
    },
    {
        "question": "I forgot my password. What should I do?",
        "answer": "Select Forgot Password on the login page and follow the password reset instructions."
    },
    {
        "question": "Can I shop without creating an account?",
        "answer": "Guest checkout may be available, depending on the store's ordering options."
    },
    {
        "question": "Do you deliver internationally?",
        "answer": "International delivery availability depends on the store's shipping destinations."
    },
    {
        "question": "How can I contact customer support?",
        "answer": "Visit the Help or Contact Us section on the website for available support options."
    },
    {
        "question": "Are products covered by a warranty?",
        "answer": "Warranty coverage depends on the product and its warranty terms."
    },
    {
        "question": "How can I delete my account?",
        "answer": "Check your account settings or contact customer support to request account deletion."
    }

]


def preprocess_text(text):
    text = text.lower()
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )
    tokens = wordpunct_tokenize(text)
    return " ".join(tokens)



questions = [
    preprocess_text(faq["question"])
    for faq in faqs
]

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(questions)


def get_answer(user_question):
    processed_question = preprocess_text(user_question)

    if not processed_question.strip():
        return "Please enter a question."

    greetings = [
        "hi", "hello", "hey",
        "good morning", "good evening"
    ]

    if processed_question in greetings:
        return "Hello! Welcome to our store. How can I help you today?"

    if processed_question in [
        "thanks", "thank you", "thank you very much"
    ]:
        return "You're welcome! Happy shopping!"

    if processed_question in [
        "how are you",
        "how are you doing"
    ]:
        return "I'm doing great! 😊 How can I help you with your shopping today?"

    if processed_question in [
            "bye",
            "goodbye",
            "tata"
    ]:
        return "Bye! 😊 Have a great day!"

    if "your name" in processed_question or "who are you" in processed_question:
        return "I'm ShopAssist, your friendly online shopping assistant! 😊"

    
    user_vector = vectorizer.transform(
        [processed_question]
    )

    similarities = cosine_similarity(
        user_vector, faq_vectors
    )[0]

    best_match_index = similarities.argmax()
    best_score = similarities[best_match_index]

    if best_score < 0.30:
        return (
            "Sorry, I couldn't find a suitable answer. "
            "Please contact customer support."
        )

    return faqs[best_match_index]["answer"]


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = data.get("message", "")

    if not isinstance(question, str):
        return jsonify({"answer": "Please enter a valid question."}), 400

    answer = get_answer(question)

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
