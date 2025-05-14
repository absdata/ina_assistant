# Inna AI Assistant

Inna is a Telegram-based AI startup co-founder powered by CrewAI. She's designed to be helpful, supportive, and insightful while maintaining a unique personality that makes interactions engaging and productive.

## Features

- 🤖 **CrewAI Agents**:
  - Planner: Breaks down requests into actionable subgoals
  - Doer: Executes tasks and performs research
  - Critic: Reviews results and suggests improvements
  - Responder: Crafts engaging responses with Inna's personality

- 📝 **Message Processing**:
  - Customizable trigger names (default: "Inna" or "Ina")
  - Automatic document processing without triggers
  - Saves all messages and embeddings to Supabase
  - Supports context-aware conversations

- 📎 **Document Processing**:
  - Automatic handling of supported file types:
    - PDF documents
    - Word documents (DOCX, DOC)
    - Text files (TXT)
  - Smart document summarization
  - Content-based suggestions
  - Historical context integration
  - Vector search for relevant content

- 🧠 **Memory Systems**:
  - Short-term memory for recent interactions
  - Long-term memory for historical context
  - Entity memory for tracking important information
  - Contextual memory for adaptive responses

- 🔍 **Azure OpenAI Integration**:
  - Uses text-embedding-3-large for embeddings
  - Semantic search capabilities
  - Context-aware responses

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ina
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your configuration:
```env
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_VERSION=your_api_version
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=your_embedding_deployment
AZURE_EMBEDDING_DIMENSION=2000

# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
SUPABASE_EMBEDDING_DIMENSION=2000

# Agent Configuration
AGENT_TRIGGER_NAMES=ina,inna
AGENT_DEFAULT_NAME=Ina
```

4. Set up the Supabase database:
- Create a new Supabase project
- Run the schema.sql file to create required tables
- Enable the pgvector extension

## Usage

1. Start the bot:
```bash
python main.py
```

2. In Telegram:
- Start a conversation with your bot
- Send a message with configured trigger words (default: "Ina" or "Inna")
- Or simply send a supported document for automatic processing
- The bot will respond with relevant information and suggestions

### Document Handling

When you send a document:
1. The bot automatically detects the file type
2. Processes the content without requiring trigger words
3. Provides a summary with:
   - Main topics
   - Key points
   - Relevant historical context
   - Content-based suggestions

### Customizing Trigger Words

You can customize how users activate the bot:

1. **Using Environment Variables**:
   ```env
   # Add multiple trigger words separated by commas
   AGENT_TRIGGER_NAMES=ina,inna,assistant,bot
   
   # Set the bot's default name in responses
   AGENT_DEFAULT_NAME=Ina
   ```

2. **Direct Settings**:
   - Edit `config/settings.py` to modify `SUPPORTED_DOCUMENT_TYPES`
   - Add or modify trigger words in the configuration

## Development

### Project Structure
```
/
├── main.py                 # Entry point
├── agents/                 # CrewAI agents
│   ├── planner.py
│   ├── doer.py
│   ├── critic.py
│   └── responder.py
├── services/              # Core services
│   ├── telegram.py
│   ├── supabase.py
│   ├── embedding.py
│   └── file_handler.py
├── memory/               # Memory management
│   ├── vector_store.py
│   └── chunking.py
├── config/              # Configuration
│   ├── settings.py      # Central configuration
│   └── logging_config.py # Logging setup
└── database/           # Database schema
    └── schema.sql
```

### Adding New Features

1. **New Agent Capabilities**:
- Extend agent classes in the `agents/` directory
- Update the CrewAI workflow in `services/telegram.py`

2. **Document Support**:
- Add new document types in `config/settings.py`
- Implement handlers in `services/file_handler.py`
- Update document processing logic in agents

3. **Custom Commands**:
- Add new command handlers in `services/telegram.py`
- Implement corresponding logic in relevant services

4. **Memory Integration**:
- Extend memory systems in agent implementations
- Add new memory types for specific features
- Implement custom memory persistence strategies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 