"""
PurrMart — Demo web app for the Customer Service Bot.

Run from the project root:   python Demo/app.py
Then open:                    http://localhost:5001

Everything this demo needs lives inside Demo/ (SQLite DB + web UI). It only
*reads* the trained model + response pools from the parent project to power
the AI support chat, and never modifies them.

Demo account (seeded automatically):  demo / demo123
"""

import os
import random
import sqlite3
import sys
from functools import wraps

import torch
from flask import (Flask, g, jsonify, redirect, render_template, request,
                   session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

# ---------------------------------------------------------------------------
# Reuse the trained model + helpers from the project root (read-only)
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.model import NeuralNet                     # noqa: E402
from src.nltk_utils import get_sentence_vector      # noqa: E402
from responses import responses as RESPONSES        # noqa: E402

# ---------------------------------------------------------------------------
# Flask setup
# ---------------------------------------------------------------------------
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DEMO_DIR, "catbot.db")
MODEL_FILE = os.path.join(ROOT_DIR, "saved_models", "trained_model.pth")

app = Flask(__name__)
app.secret_key = os.urandom(24)  # session signing key (fresh each run, fine for a demo)

# ---------------------------------------------------------------------------
# Load the AI model (the exact same weights the backend API uses)
# ---------------------------------------------------------------------------
checkpoint = torch.load(MODEL_FILE)
input_size = checkpoint["input_size"]
hidden_size = checkpoint["hidden_size"]
output_size = checkpoint["output_size"]
tags = checkpoint["tags"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(checkpoint["model_state"])
model.eval()


def predict(sentence):
    """Classify a sentence -> (tag, confidence 0..1)."""
    X = get_sentence_vector(sentence)
    X = torch.from_numpy(X).float().unsqueeze(0)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()].item()
    return tags[predicted.item()], prob


# ---------------------------------------------------------------------------
# Shop catalogue (cat food)
# ---------------------------------------------------------------------------
PRODUCTS = [
    {"id": 1, "name": "Purrfect Salmon Feast", "emoji": "🐟", "price": 12.99,
     "desc": "Wild salmon with omega-3 for a shiny coat."},
    {"id": 2, "name": "Tuna & Tofu Delight", "emoji": "🍣", "price": 9.49,
     "desc": "Light tuna pâté with plant protein."},
    {"id": 3, "name": "Chicken Casserole Bites", "emoji": "🍗", "price": 11.20,
     "desc": "Slow-cooked chicken chunks in rich gravy."},
    {"id": 4, "name": "Whisker Wellness Kibble", "emoji": "🥣", "price": 15.75,
     "desc": "Complete everyday nutrition in crunchy bites."},
    {"id": 5, "name": "Senior Cat Comfort Meal", "emoji": "😌", "price": 13.40,
     "desc": "Gentle, easy-to-digest formula for older cats."},
    {"id": 6, "name": "Kitten Growth Formula", "emoji": "🐱", "price": 14.10,
     "desc": "Protein-rich kibble for playful growing kittens."},
    {"id": 7, "name": "Grain-Free Gourmet Pâté", "emoji": "👑", "price": 16.30,
     "desc": "Premium pâté fit for a royal cat."},
    {"id": 8, "name": "Hairball Control Crunch", "emoji": "🧶", "price": 10.60,
     "desc": "Crispy bites that help reduce hairballs."},
]

# ---------------------------------------------------------------------------
# Database (SQLite)
# ---------------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            balance       REAL NOT NULL DEFAULT 3000.0
        );
        CREATE TABLE IF NOT EXISTS orders (
            id           TEXT PRIMARY KEY,
            user_id      INTEGER NOT NULL,
            product_name TEXT NOT NULL,
            price        REAL NOT NULL,
            status       TEXT NOT NULL DEFAULT 'active',
            created_at   TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    # Seed a demo account so the page can be tried immediately.
    if db.execute("SELECT 1 FROM users WHERE username = 'demo'").fetchone() is None:
        db.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                   ("demo", generate_password_hash("demo123")))
    db.commit()
    db.close()


def current_user():
    if "user_id" not in session:
        return None
    return get_db().execute(
        "SELECT * FROM users WHERE id = ?", (session["user_id"],)
    ).fetchone()


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            if request.path.startswith("/api/"):
                return jsonify({"error": "Not logged in"}), 401
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=current_user())


@app.route("/support")
@login_required
def support():
    return render_template("support.html", user=current_user())


# ---------------------------------------------------------------------------
# Auth API
# ---------------------------------------------------------------------------
@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json(force=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters"}), 400
    if len(password) < 4:
        return jsonify({"error": "Password must be at least 4 characters"}), 400

    db = get_db()
    if db.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
        return jsonify({"error": "Username already taken"}), 409

    db.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
               (username, generate_password_hash(password)))
    db.commit()
    uid = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()["id"]
    session["user_id"] = uid
    return jsonify({"ok": True, "username": username})


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(force=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    row = get_db().execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if row is None or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "Invalid username or password"}), 401

    session["user_id"] = row["id"]
    return jsonify({"ok": True, "username": row["username"], "balance": row["balance"]})


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/me")
@login_required
def me():
    user = current_user()
    return jsonify({"username": user["username"], "balance": user["balance"]})


# ---------------------------------------------------------------------------
# Shop API
# ---------------------------------------------------------------------------
@app.route("/api/products")
def list_products():
    return jsonify(PRODUCTS)


@app.route("/api/order", methods=["POST"])
@login_required
def place_order():
    user = current_user()
    data = request.get_json(force=True) or {}
    product = next((p for p in PRODUCTS if str(p["id"]) == str(data.get("product_id"))), None)
    if product is None:
        return jsonify({"error": "Unknown product"}), 400

    price = product["price"]
    if user["balance"] < price:
        return jsonify({"error": "Insufficient balance"}), 400

    db = get_db()
    # Generate a unique order number like CB-482913
    while True:
        order_id = f"CB-{random.randint(100000, 999999)}"
        if db.execute("SELECT 1 FROM orders WHERE id = ?", (order_id,)).fetchone() is None:
            break

    new_balance = round(user["balance"] - price, 2)
    db.execute("INSERT INTO orders (id, user_id, product_name, price) VALUES (?, ?, ?, ?)",
               (order_id, user["id"], product["name"], price))
    db.execute("UPDATE users SET balance = ? WHERE id = ?", (new_balance, user["id"]))
    db.commit()

    return jsonify({
        "order_id": order_id,
        "product": product["name"],
        "price": price,
        "balance": new_balance,
    })


@app.route("/api/orders")
@login_required
def list_orders():
    user = current_user()
    rows = get_db().execute(
        """SELECT id, product_name, price, status, created_at
           FROM orders WHERE user_id = ? ORDER BY created_at DESC, id DESC""",
        (user["id"],),
    ).fetchall()
    return jsonify([dict(r) for r in rows])


# ---------------------------------------------------------------------------
# AI Support Chat
# ---------------------------------------------------------------------------
# user_id -> intent that is waiting for an order ID to be supplied
PENDING_ORDER_ID = {}

INTENT_NEEDS_ORDER = {"Return_Refund", "Cancel_Order", "Order_Status"}

ORDER_STATUS_TEXT = {
    "active": "Active 🚚 — your order is on its way",
    "refunded": "Refunded 💸 — the money is back in your account",
    "cancelled": "Cancelled ❌ — this order was cancelled",
}

CHAT_STOP_WORDS = {"cancel", "stop", "nevermind", "never mind", "exit", "quit", "forget it"}


@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    user = current_user()
    message = (request.get_json(force=True) or {}).get("message", "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    # If the bot is waiting for an order ID, the next message is treated as one.
    waiting = PENDING_ORDER_ID.get(user["id"])
    if waiting is not None:
        if message.lower() in CHAT_STOP_WORDS:
            PENDING_ORDER_ID.pop(user["id"], None)
            return jsonify({"tag": waiting, "confidence": "-",
                            "response": "No problem — I've dropped that request. Anything else I can help with? 😺"})
        return jsonify(handle_order_id(user, waiting, message))

    tag, prob = predict(message)
    confidence = f"{prob * 100:.2f}%"

    # Not confident enough -> fallback response (same as the backend API).
    if prob < 0.7:
        return jsonify({"tag": "Unknown", "confidence": confidence,
                        "response": "I'm sorry, I didn't quite understand that. Could you rephrase?"})

    if tag in INTENT_NEEDS_ORDER:
        PENDING_ORDER_ID[user["id"]] = tag
        return jsonify({"tag": tag, "confidence": confidence,
                        "response": f"{random.choice(RESPONSES[tag])} Please send me your order ID (e.g. CB-123456)."})

    if tag == "Payment_Issue":
        return jsonify({"tag": tag, "confidence": confidence,
                        "response": random.choice(RESPONSES[tag]) +
                                    " Don't worry — no money is taken for unconfirmed orders, so your balance is safe."})

    if tag == "Complaint":
        return jsonify({"tag": tag, "confidence": confidence,
                        "response": "We take complaints very seriously. Could you tell me more about what went wrong?"})

    if tag == "Speak_To_Human":
        return jsonify({"tag": tag, "confidence": confidence,
                        "response": random.choice(RESPONSES[tag]) +
                                    " A member of our team will reach out to you shortly. Meanwhile, is there anything else I can help with?"})

    # Bot_Identity
    return jsonify({"tag": tag, "confidence": confidence,
                    "response": random.choice(RESPONSES[tag])})


def handle_order_id(user, intent, raw):
    """Process an order ID supplied during a Return_Refund / Cancel_Order / Order_Status flow."""
    order_id = raw.strip().upper()
    db = get_db()
    order = db.execute(
        "SELECT * FROM orders WHERE id = ? AND user_id = ?", (order_id, user["id"])
    ).fetchone()

    if order is None:
        return {"tag": intent, "confidence": "-",
                "response": "Hmm, I couldn't find an order with that ID in your account. "
                            "Double-check the number and send it again, or type 'cancel' to stop."}

    PENDING_ORDER_ID.pop(user["id"], None)  # flow completed
    product, price, status = order["product_name"], order["price"], order["status"]

    if intent == "Order_Status":
        return {"tag": intent, "confidence": "-",
                "response": f"Order {order_id} — {product} (${price:.2f}) placed on {order['created_at']}.\n"
                            f"Status: {ORDER_STATUS_TEXT.get(status, status)}."}

    if intent == "Return_Refund":
        if status == "active":
            new_balance = round(user["balance"] + price, 2)
            db.execute("UPDATE orders SET status = 'refunded' WHERE id = ?", (order_id,))
            db.execute("UPDATE users SET balance = ? WHERE id = ?", (new_balance, user["id"]))
            db.commit()
            return {"tag": intent, "confidence": "-",
                    "response": f"Refund approved! 💸 ${price:.2f} for '{product}' has been returned to your account. "
                                f"Your new balance is ${new_balance:.2f}."}
        return {"tag": intent, "confidence": "-",
                "response": f"This order is already {status}, so there's nothing left to refund."}

    # Cancel_Order
    if status == "active":
        new_balance = round(user["balance"] + price, 2)
        db.execute("UPDATE orders SET status = 'cancelled' WHERE id = ?", (order_id,))
        db.execute("UPDATE users SET balance = ? WHERE id = ?", (new_balance, user["id"]))
        db.commit()
        return {"tag": intent, "confidence": "-",
                "response": f"Order {order_id} has been cancelled. ❌ ${price:.2f} was credited back — "
                            f"your new balance is ${new_balance:.2f}."}
    return {"tag": intent, "confidence": "-",
            "response": f"This order is already {status}, so it can't be cancelled."}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    print("PurrMart demo running at http://localhost:5001")
    app.run(debug=True, port=5001)
