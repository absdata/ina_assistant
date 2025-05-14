-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Messages table
CREATE TABLE IF NOT EXISTS ina_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id BIGINT NOT NULL,
    chat_id BIGINT NOT NULL,
    message_text TEXT NOT NULL,
    file_content TEXT,
    file_name TEXT,
    file_type TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Message embeddings table
CREATE TABLE IF NOT EXISTS ina_message_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID REFERENCES ina_messages(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding vector(2000) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_messages_user_id ON ina_messages(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON ina_messages(chat_id);
CREATE INDEX IF NOT EXISTS idx_message_embeddings_message_id ON ina_message_embeddings(message_id);

-- Create a function to update updated_at timestamp
CREATE OR REPLACE FUNCTION ina_update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create a trigger to automatically update updated_at
CREATE TRIGGER ina_update_messages_updated_at
    BEFORE UPDATE ON ina_messages
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column(); 