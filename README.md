# Requirements Analyst Agent (LangChain + Ollama)

Analyze e-commerce user stories and extract testable requirements using LangChain with a local Ollama model. The agent can output fast, structured TEXT analysis and convert it to a strict JSON schema for assignment submissions.

## 🚀 Features

- **Fast structured analysis (TEXT)**: Clearly formatted sections for requirements.
- **JSON conversion**: Convert TEXT output into the assignment's JSON schema with IDs and defaults.
- **Robust handling**: Retries, timeouts, deterministic fallback, and flexible parsing for bullet styles.
- **Edge-case tests**: Built-in test runner to validate input handling.

## 📋 Prerequisites

- Python 3.10+
- Ollama installed and running locally (with the `llama3` model pulled)
- Git (optional)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Assignment1
```

### 2. Install Ollama and pull model

Install Ollama from `https://ollama.com`. After installation:

```bash
# Start Ollama service (if not already running)
ollama serve

# Pull an LLM model (recommended)
ollama pull llama3
```

### 3. Create Virtual Environment

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. (Optional) Environment Variables

Copy the template if you plan to add API keys later:

```bash
# Windows
copy env_template.txt .env

# macOS/Linux
cp env_template.txt .env
```

## 🎯 Usage

### Basic Usage

Run the main script to analyze a sample e-commerce user story using Ollama locally:

```bash
python main.py
```

This runs a built-in demo that extracts functional/non-functional requirements, edge cases, and gaps. It prints the TEXT analysis, the converted JSON, and saves both to files.

### Programmatic Usage

```python
from main import RequirementsAnalyst

analyst = RequirementsAnalyst()
user_story = "As a customer, I want to add products to my shopping cart and modify quantities, so that I can purchase multiple items in a single order."
acceptance_criteria = """
- User can add products to cart from product detail page
- User can update item quantities in the cart
- User can remove items from the cart
- Cart displays total price including tax
- Cart persists during the user session
- Maximum 10 items of any single product allowed
- Out-of-stock items cannot be added to cart
- Cart shows item availability status
- User receives confirmation when items are added/removed
"""
print(analyst.analyze_user_story(user_story, acceptance_criteria))
```

Convert TEXT to JSON:

```python
text = analyst.analyze_user_story(user_story, acceptance_criteria)
json_result = analyst.text_to_json_converter(text, user_story)
```

```python
from main import RequirementsAnalystAgent

# Initialize the agent
agent = RequirementsAnalystAgent()

# Analyze your requirements
user_input = """
I need a mobile app for food delivery that allows:
- Users to browse restaurants and menus
- Place orders with customizations
- Track delivery in real-time
- Rate and review restaurants
- Payment processing
"""

# Get comprehensive analysis
result = agent.analyze_requirements(user_input)

# Generate formatted document
document = agent.generate_requirements_document(result)
print(document)
```

### Example Output (abridged)

Text:

```
FUNCTIONAL REQUIREMENTS:
- Add products to cart from product detail page
- Update item quantities in the cart

NON-FUNCTIONAL REQUIREMENTS:
- Response time < 3 seconds for cart actions

EDGE CASES:
- Adding more than 10 items

GAPS IDENTIFIED:
- Authentication policy not specified
```

JSON:

```json
{
  "user_story": "As a customer, I want to add products to my shopping cart and modify quantities...",
  "functional_requirements": [
    {"id": "FR001", "description": "Add products to cart from product detail page", "priority": "High", "category": "Core Functionality", "testable": true}
  ],
  "non_functional_requirements": [
    {"id": "NFR001", "description": "Response time < 3 seconds for cart actions", "type": "Performance", "measurable": true}
  ],
  "edge_cases": [
    {"id": "EC001", "description": "Adding more than 10 items", "scenario": "Adding more than 10 items"}
  ],
  "gaps_identified": [
    "Authentication policy not specified"
  ]
}
```

## 📁 Project Structure

```
Assignment1/
├── main.py            # LangChain + Ollama analyst (single-file implementation)
├── requirements.txt   # Python dependencies
├── env_template.txt   # Optional environment template
└── README.md          # Setup and usage
```

## 🔧 Configuration

- Default model: `llama3` via local Ollama
- Configure via environment variables (or pass `AnalystConfig` in code):
  - `OLLAMA_MODEL` (default: `llama3`)
  - `LLM_TIMEOUT_SECONDS` (default: `45`)
  - `MIN_STORY_LEN` (default: `20`)
  - `MIN_ACCEPTANCE_LEN` (default: `10`)

## 🧩 JSON Schema (Assignment)

Top-level keys:
- `user_story` (string)
- `functional_requirements` (array of objects)
- `non_functional_requirements` (array of objects)
- `edge_cases` (array of objects)
- `gaps_identified` (array of strings)

Functional requirement object:
- `id` (FR001, FR002, ...)
- `description`
- `priority` (default: High)
- `category` (default: Core Functionality)
- `testable` (default: true)

Non-functional requirement object:
- `id` (NFR001, NFR002, ...)
- `description`
- `type` (Performance, Security, Scalability, Reliability, Usability, or Unspecified)
- `measurable` (default: true)

Edge case object:
- `id` (EC001, EC002, ...)
- `description`
- `scenario`

## 🧪 API Reference

All methods below are in `main.py` within `RequirementsAnalyst`.

- `analyze_user_story(user_story: str, acceptance_criteria: str) -> str`
  - Returns TEXT analysis with four sections. Handles input validation, retries, and timeouts.

- `text_to_json_converter(analysis_text: str, original_user_story: str) -> dict`
  - Parses TEXT into JSON schema. Supports headers with `:` or `-`, and bullets `•`, `-`, `*`. Applies defaults.

- `_parse_plain_text_sections(text: str) -> dict`
  - Internal: splits text into sections and bullet lines.

- `_infer_nfr_type(description: str) -> str`
  - Internal: infers NFR type from keywords.

- `_json_to_text(data: dict) -> str`
  - Internal: converts JSON back to TEXT sections.

- `validate_assignment_requirements(data: dict) -> dict`
  - Validates at least 10 total items, ID formats, and schema completeness.

### CLI Test Runner
- Run edge tests: `python -X utf8 main.py --test`
  - Tests empty, very short, malformed AC, very long, and emojis/special characters.

## 🛠️ Troubleshooting

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

1. Ensure Ollama is running: `ollama serve` (or background service)
2. Pull the model: `ollama pull llama3`
3. Reinstall deps if imports fail: `pip install -r requirements.txt`
4. Windows encoding issues: run with `python -X utf8 main.py`.
5. Empty outputs: ensure the model is running; deterministic fallback will still produce text and JSON, but counts may be low.
6. Bullet parsing: use headings with `:` or `-`, and bullets `•`, `-`, or `*`.

## ✅ Validation Criteria (Assignment)

Your submission should satisfy:
- At least 10 total requirements across functional, non-functional, edge cases, and gaps
- Proper ID formats: `FR###`, `NFR###`, `EC###`
- All required fields present per schema
- Valid, parsable JSON (for the JSON output)

## 🔗 Dependencies

- `langchain`
- `langchain-ollama`
- `python-dotenv`

## 📈 Future Enhancements

- [ ] Web interface for easier interaction
- [ ] Support for multiple LLM providers
- [ ] Export to different document formats (PDF, Word, etc.)
- [ ] Integration with project management tools
- [ ] Custom agent training capabilities
- [ ] Real-time collaboration features

---

**Note**: This project runs fully locally with Ollama (no API key required).

## 📚 Resources

- [CrewAI Documentation](https://docs.crewai.com/) – Multi-agent orchestration concepts and patterns
- [LangChain Documentation](https://python.langchain.com/) – Chains, prompts, and LLM integration patterns
- [Ollama Documentation](https://ollama.com/docs) – Local LLM runtime, models, and serving

## 🧪 JSON Schema Validation Tools

- Python: [jsonschema](https://python-jsonschema.readthedocs.io/) – Validate JSON instance against a schema
- JavaScript/TypeScript: [Ajv](https://ajv.js.org/) – High-performance JSON schema validator
- Online validators: [JSON Schema Validator](https://www.jsonschemavalidator.net/), [JSONLint](https://jsonlint.com/)
- Schema authoring: [JSON Schema Draft 2020-12](https://json-schema.org/)

## ✅ Requirements Engineering Best Practices

- Use MoSCoW prioritization (Must/Should/Could/Won’t) for stakeholder alignment
- Maintain traceability from user stories → requirements → test cases → defects
- Ensure each requirement is SMART (Specific, Measurable, Achievable, Relevant, Time-bound)
- Separate functional vs non-functional requirements; make NFRs measurable (e.g., response time, uptime)
- Capture edge cases and negative scenarios early; design for validation and error handling
- Reference standards: [IEEE 29148 Requirements Engineering](https://ieeexplore.ieee.org/document/7100875) and [BABOK](https://www.iiba.org/babok-guide/)

## 🧠 NLP Techniques for Business Analysis

- Keyword and keyphrase extraction (e.g., RAKE, TextRank) for initial signal
- Named Entity Recognition (NER) to identify actors, products, locations
- Dependency parsing and chunking to extract action-object structures ("add product", "update quantity")
- Pattern-based extraction for constraints (max/min, limits, response times)
- Classification to separate Functional vs Non-Functional requirements
- Similarity search to detect duplicates/overlaps and map to acceptance criteria

## 🧩 Sample User Stories for Additional Testing

- Search functionality
  - As a shopper, I want to search for products by name and category so that I can quickly find items of interest.
  - AC:
    - Results include product name, price, and availability
    - Support partial matches and common typos
    - Filter by category and price range

- Checkout and payments
  - As a customer, I want to securely pay for my items so that I can complete the purchase.
  - AC:
    - Support credit/debit cards and digital wallets
    - Show final total with tax and shipping before payment
    - Payment confirmation sent via email

- Wishlist
  - As a registered user, I want to add items to a wishlist so that I can purchase them later.
  - AC:
    - Add/remove items from wishlist
    - Wishlist persists across sessions
    - Notify if wishlist item goes on sale

