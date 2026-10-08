# Excel Data Analysis Agent

Ask questions in plain English about an Excel workbook. The agent loads the first worksheet, analyzes its columns, and answers using generated pandas code. It includes a Streamlit chat interface and an interactive command-line interface, plus a built-in inventory and finance glossary.

## Requirements

- Python 3.10 or newer
- An Excel workbook in `.xlsx` format (the sample workbook is included)
- API credentials are optional; without them, the agent uses its built-in rule-based analysis

## Setup

Create and activate a virtual environment, then install the core packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas openpyxl streamlit python-dotenv
```

To enable LLM-assisted code generation and summaries, install one or both providers:

```powershell
python -m pip install anthropic openai
```

Create a `.env` file in the project root and add the key for the provider you want to use:

```dotenv
ANTHROPIC_API_KEY=your-anthropic-api-key
GROQ_API_KEY=your-groq-api-key
```

You only need to set a key for a provider you installed. The agent tries Anthropic first, then Groq. Keep `.env` private and never commit API keys.

## Run the Streamlit app

```powershell
streamlit run app_streamlit.py
```

The app opens with the included inventory workbook. Use the sidebar to upload an `.xlsx` or `.xls` workbook, and ask questions in the chat. For `.xls` files, install the additional Excel reader:

```powershell
python -m pip install xlrd
```

## Run the command-line chat

Start with the sample workbook:

```powershell
python run_excel_chat.py
```

Use a different workbook by passing its path:

```powershell
python run_excel_chat.py "path/to/workbook.xlsx"
```

Useful options:

```powershell
python run_excel_chat.py --demo 3 --non-interactive
python run_excel_chat.py --verbose
```

In interactive mode, type `sample` to list suggested questions, `columns` to inspect the dataset schema, `summary` to view conversation statistics, or `quit` to exit. The `verbose` command toggles generated code and raw output.

## Example questions

- What is the total inventory value of all products?
- Show me the top 5 products by number of units sold.
- Which products have more than 70 units on hand?
- Compare Laptop vs Monitor.
- Define inventory turnover.
- Give me a summary of the dataset.

## Project files

- `app_streamlit.py`: Streamlit user interface
- `run_excel_chat.py`: command-line interface
- `excel_agent/agent.py`: query classification, analysis, and response generation
- `excel_agent/data_loader.py`: workbook and worksheet loading
- `excel_agent/code_executor.py`: pandas code execution
- `excel_agent/search_tool.py`: built-in inventory and finance definitions
- `Inventory-Records-Sample-Data.xlsx`: sample workbook

## Data and credentials

When an LLM provider is configured, the agent sends the user's question, workbook title, column names and sample values, and analysis output to that provider to generate code or summaries. Do not use LLM mode with data you are not permitted to share. The generated-code executor is not a security sandbox; only analyze workbooks you trust. If an API key has been exposed, revoke it and create a replacement.