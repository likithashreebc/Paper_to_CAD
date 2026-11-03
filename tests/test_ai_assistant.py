import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "pipeline"))

from ai_assistant import AICircuitAssistant
from circuit_data_model import CircuitGraph, ComponentModel
from circuit_validator import ValidationFinding


class TestAICircuitAssistantCalculations(unittest.TestCase):
    def setUp(self):
        dotenv_patcher = patch("dotenv.load_dotenv")
        dotenv_patcher.start()
        self.addCleanup(dotenv_patcher.stop)
        self.graph = CircuitGraph(
            image_name="calculation_test",
            width=100,
            height=100,
            components=[
                ComponentModel(id=0, type="resistor"),
                ComponentModel(id=1, type="capacitor.unpolarized"),
                ComponentModel(id=2, type="inductor"),
            ],
        )
        self.assistant = AICircuitAssistant(api_key="none")

    def test_calculates_resistance_from_explicit_voltage_and_current(self):
        answer = self.assistant.explain_circuit(
            self.graph, user_question="Calculate R1 from V=5V and I=10mA"
        )

        self.assertIn("500 ohm", answer)
        self.assertIn("R1 = 500", answer)
        self.assertIn("R = V / I = 5 V / 0.01 A", answer)
        self.assertIn("Use Edit to apply it", answer)

    def test_calculates_capacitance_from_time_constant_and_resistance(self):
        answer = self.assistant.explain_circuit(
            self.graph, user_question="Calculate C1 from tau=2ms and R=10k"
        )

        self.assertIn("200 nF", answer)
        self.assertIn("C = tau / R", answer)

    def test_requests_missing_inputs_instead_of_guessing(self):
        answer = self.assistant.explain_circuit(self.graph, user_question="Calculate L1")

        self.assertIn("time constant and resistance", answer)
        self.assertIn("will not guess", answer)

    def test_offline_overview_answers_question_and_includes_refined_component_types(self):
        answer = self.assistant.explain_circuit(
            self.graph, user_question="What does this circuit do?"
        )

        self.assertIn("capacitor.unpolarized 1", answer)
        self.assertIn("not a full conversational analysis", answer)
        self.assertNotIn("console.groq.com", answer)

    def test_offline_inventory_lists_every_detected_component(self):
        answer = self.assistant.explain_circuit(
            self.graph, user_question="What components are detected?"
        )

        self.assertIn("Component 0: resistor", answer)
        self.assertIn("Component 1: capacitor.unpolarized", answer)
        self.assertIn("Component 2: inductor", answer)

    def test_offline_engine_says_when_open_ended_question_is_unsupported(self):
        answer = self.assistant.explain_circuit(
            self.graph, user_question="Explain the history of electrical engineering"
        )

        self.assertIn("offline grounded mode", answer)
        self.assertIn("can’t handle open-ended questions", answer)

    def test_warning_query_returns_validation_evidence_and_action(self):
        finding = ValidationFinding(
            rule_id="RULE-GND-001",
            severity="warning",
            title="Missing Ground Reference",
            evidence="No ground symbol was identified.",
            explanation="A ground reference is needed.",
            suggested_action="Add an explicit ground symbol.",
        )

        answer = self.assistant.explain_circuit(
            self.graph, findings=[finding], user_question="What is wrong with this circuit?"
        )

        self.assertIn("Missing Ground Reference", answer)
        self.assertIn("No ground symbol was identified.", answer)
        self.assertIn("Add an explicit ground symbol.", answer)

    @patch.dict(
        "os.environ",
        {"LLM_BASE_URL": "http://localhost:11434/v1", "LLM_MODEL": "local-model"},
        clear=True,
    )
    def test_local_openai_compatible_model_needs_no_api_key(self):
        assistant = AICircuitAssistant()

        self.assertIsNone(assistant.api_key)
        self.assertEqual(assistant.base_url, "http://localhost:11434/v1")
        self.assertEqual(assistant.model, "local-model")
        self.assertEqual(assistant.provider, "OpenAI-compatible LLM")

    @patch("ai_assistant.requests.post")
    @patch.dict(
        "os.environ",
        {"LLM_BASE_URL": "http://localhost:11434/v1", "LLM_MODEL": "local-model"},
        clear=True,
    )
    def test_local_llm_request_omits_auth_header(self, post):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "Local model response."}}]
        }
        post.return_value = response
        assistant = AICircuitAssistant()

        answer = assistant.explain_circuit(self.graph, user_question="Explain this circuit")

        self.assertIn("Local model response", answer)
        self.assertNotIn("Authorization", post.call_args.kwargs["headers"])

    @patch("ai_assistant.requests.post")
    def test_llm_request_uses_configured_bearer_token(self, post):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "The circuit has a resistor."}}]
        }
        post.return_value = response
        assistant = AICircuitAssistant(
            api_key="test-api-key",
            base_url="https://llm.example/v1",
            model="test-model",
        )

        answer = assistant._call_llm(self.graph, [], None, "Explain this circuit")

        self.assertIn("resistor", answer)
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer test-api-key",
        )


    def test_api_key_mistaken_for_model_is_never_used_as_model(self):
        assistant = AICircuitAssistant(
            api_key="test-api-key",
            base_url="https://api.openai.com/v1",
            model="sk-proj-abcdefghijklmnopqrstuvwxyz0123456789",
        )

        self.assertEqual(assistant.model, "gpt-4o-mini")

    @patch.dict(
        "os.environ",
        {
            "LLM_BASE_URL": "https://api.openai.com/v1",
            "LLM_MODEL": "gpt-4o-mini",
            "OPENAI_API_KEY": "openai-test-key",
            "GROQ_API_KEY": "groq-test-key",
        },
        clear=True,
    )
    def test_openai_endpoint_uses_openai_key_when_both_provider_keys_exist(self):
        assistant = AICircuitAssistant()

        self.assertEqual(assistant.provider, "OpenAI")
        self.assertEqual(assistant.base_url, "https://api.openai.com/v1")
        self.assertEqual(assistant.api_key, "openai-test-key")

    @patch("ai_assistant.requests.post")
    @patch.dict(
        "os.environ",
        {
            "LLM_BASE_URL": "https://api.openai.com/v1",
            "LLM_MODEL": "gpt-4o-mini",
            "OPENAI_API_KEY": "openai-test-key",
            "GROQ_API_KEY": "groq-test-key",
        },
        clear=True,
    )
    def test_openai_does_not_fall_back_to_groq_models(self, post):
        response = Mock()
        response.status_code = 400
        response.text = '{"error":{"message":"invalid model ID"}}'
        post.return_value = response
        assistant = AICircuitAssistant()

        with self.assertRaisesRegex(RuntimeError, "OpenAI request failed for configured model 'gpt-4o-mini'"):
            assistant._call_llm(self.graph, [], None, "Explain this circuit")

        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.kwargs["json"]["model"], "gpt-4o-mini")

    @patch.dict(
        "os.environ",
        {
            "LLM_BASE_URL": "https://api.groq.com/openai/v1",
            "LLM_MODEL": "llama-3.3-70b-versatile",
            "OPENAI_API_KEY": "openai-test-key",
            "GROQ_API_KEY": "groq-test-key",
        },
        clear=True,
    )
    def test_groq_endpoint_uses_groq_key_when_both_provider_keys_exist(self):
        assistant = AICircuitAssistant()

        self.assertEqual(assistant.provider, "Groq")
        self.assertEqual(assistant.base_url, "https://api.groq.com/openai/v1")
        self.assertEqual(assistant.api_key, "groq-test-key")


if __name__ == "__main__":
    unittest.main()
