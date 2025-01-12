from phi.agent import Agent
from phi.model.groq import Groq
from phi.embedder.google import GeminiEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2, SearchType
from phi.tools.exa import ExaTools

import os
from dotenv import load_dotenv

load_dotenv()
embedder = GeminiEmbedder(model="models/text-embedding-004", dimensions=512)

db_url="postgresql+psycopg://ai:ai@localhost:5532/ai"
# Create a knowledge base from a PDF
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    # Use LanceDB as the vector database
    vector_db=PgVector2(
        collection="recipes",
        db_url=db_url,
      #  search_type=SearchType.vector,
        embedder=embedder
    ),
)
# Comment out after first run as the knowledge base is loaded
#knowledge_base.load(recreate=False)

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    # Add the knowledge base to the agent
    knowledge=knowledge_base,
    search_knowledge=True,
    tools=[ExaTools()],
    show_tool_calls=True,
    markdown=True,
    debug=True,
    instructions=[
        "Search for recipes based on the ingredients and time available from the knowledge base.",
        "Include the exact calories, preparation time, cooking instructions, and highlight allergens for the recommended recipes.",
        "If the recipe is not available in knowledge base say Im sorry i dont know the recipe",
        "Provide a list of recipes that match the user's requirements and preferences.",
        #"generate the response from knowlagbe base only"
    ],
)
agent.print_response("How do I make Som Tum", stream=True)
