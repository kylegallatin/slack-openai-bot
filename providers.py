from abc import ABC, abstractmethod
from typing import Any, Dict, List

from google import genai
from google.genai import types
from openai import OpenAI
from openai.types.responses.response_output_message import ResponseOutputMessage

class AIProvider(ABC):
    @abstractmethod
    def generate(
        self,
        input_text: str,
        tools: List[Dict[str, Any]],
        instructions: str,
    ) -> str:
        """Generate a response from the AI model.
        
        Args:
            input_text: The input text to generate a response for
            tools: List of tool definitions for function calling
            instructions: System instructions to guide the model
            
        Returns:
            Generated text response
        """
        pass

class OpenAIProvider(AIProvider):
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def _process_response_output(self, output: List) -> str:
        for response_object in output:
            if type(response_object) == ResponseOutputMessage:
                return response_object.content[0].text
        return ""

    def generate(
        self,
        input_text: str,
        tools: List[Dict[str, Any]],
        instructions: str,
    ) -> str:
        response = self.client.responses.create(
            model="gpt-4",
            tools=tools,
            tool_choice="auto",
            input=input_text,
            instructions=instructions
        )
        return self._process_response_output(response.output)

class VertexProvider(AIProvider):
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )
        self.model = "gemini-2.0-flash-exp"

    def generate(
        self,
        input_text: str,
        tools: List[Dict[str, Any]],
        instructions: str,
    ) -> str:
        # Convert tools to Vertex AI format
        vertex_tools = []
        for tool in tools:
            vertex_tools.append(types.Tool(**tool))

        generate_content_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=8192,
            response_modalities=["TEXT"],
            tools=vertex_tools,
        )

        # Create contents with instructions and input
        contents = [
            types.Part.from_text(text=instructions),
            types.Part.from_text(text=input_text)
        ]

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=generate_content_config,
        )

        # Return the text directly
        if not response.candidates or not response.candidates[0].content.parts:
            return ""
            
        return response.candidates[0].content.parts[0].text