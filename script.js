const API_URL = "http://127.0.0.1:5000/tcs";

const form = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const messages = document.getElementById("messages");
const welcome = document.getElementById("welcome");
const sendButton = document.getElementById("send-btn");
const newChatButton = document.getElementById("new-chat");

function addMessage(role, text) {
    const message = document.createElement("div");
    message.className = "message";

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "👤" : "✦";

    const content = document.createElement("div");
    content.className = "message-content";

    const heading = document.createElement("strong");
    heading.textContent = role === "user" ? "You" : "TCS AI";

    const body = document.createElement("div");
    body.textContent = text;

    content.appendChild(heading);
    content.appendChild(body);
    message.appendChild(avatar);
    message.appendChild(content);
    messages.appendChild(message);

    welcome.style.display = "none";
    messages.scrollIntoView({ block: "end", behavior: "smooth" });

    return body;
}

async function askQuestion(question) {
    const cleanQuestion = question.trim();

    if (!cleanQuestion || sendButton.disabled) {
        return;
    }

    addMessage("user", cleanQuestion);
    questionInput.value = "";
    questionInput.style.height = "auto";
    sendButton.disabled = true;
    sendButton.textContent = "...";

    const answerElement = addMessage("assistant", "Thinking...");

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                tcs_question: cleanQuestion
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "The server returned an error.");
        }

        answerElement.textContent = data.gpt_response || "No answer was returned.";

        if (data.sources && data.sources.length > 0) {
            const sourceText = data.sources.map(source => {
                const page = source.page;
                return page !== undefined && page !== null
                    ? `Page ${Number(page) + 1}`
                    : "Document source";
            });

            const uniqueSources = [...new Set(sourceText)];

            const sourceNote = document.createElement("p");
            sourceNote.textContent = "Sources: " + uniqueSources.join(", ");
            sourceNote.style.color = "#aaa";
            sourceNote.style.fontSize = "12px";
            answerElement.appendChild(sourceNote);
        }
    } catch (error) {
        answerElement.textContent =
            "Could not connect to the chatbot backend. Make sure app.py is running.\n\n" +
            "Details: " + error.message;
    } finally {
        sendButton.disabled = false;
        sendButton.textContent = "➤";
        questionInput.focus();
    }
}

form.addEventListener("submit", function(event) {
    event.preventDefault();
    askQuestion(questionInput.value);
});

questionInput.addEventListener("input", function() {
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 160) + "px";
});

questionInput.addEventListener("keydown", function(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
    }
});

document.querySelectorAll(".suggestion").forEach(button => {
    button.addEventListener("click", function() {
        askQuestion(this.dataset.question);
    });
});

newChatButton.addEventListener("click", function() {
    messages.replaceChildren();
    welcome.style.display = "block";
    questionInput.value = "";
    questionInput.focus();
});
