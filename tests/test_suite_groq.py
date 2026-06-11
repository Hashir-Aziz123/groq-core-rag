import time
import json
from src.groq_client import GroqClient

class GroqTestSuite:
    def __init__(self):
        print("[INIT] Initializing Groq API Stress Test Suite...")
        self.client = GroqClient()
        self.results = []

    def log_result(self, test_id: int, name: str, success: bool, execution_time: float, details: str):
        self.results.append({
            "id": test_id,
            "name": name,
            "success": success,
            "time": f"{execution_time:.2f}s",
            "details": details
        })

    def run_all(self):
        tests = [
            self.test_sanity_check,
            self.test_system_prompt_adherence,
            self.test_deterministic_temperature,
            self.test_high_temperature_variance,
            self.test_json_structure_generation,
            self.test_large_payload_handling,
            self.test_unicode_and_math_symbols,
            self.test_strict_token_limitation,
            self.test_consecutive_rapid_fire,
            self.test_malformed_payload_resilience
        ]
        
        print(f"[RUNNING] Executing {len(tests)} integration test cases...\n")
        for idx, test_fn in enumerate(tests, 1):
            start = time.time()
            try:
                success, details = test_fn()
                duration = time.time() - start
                self.log_result(idx, test_fn.__name__, success, duration, details)
                status = "PASSED" if success else "FAILED"
                print(f"Test {idx}/10: {test_fn.__name__} -> {status} ({duration:.2f}s)")
            except Exception as e:
                duration = time.time() - start
                self.log_result(idx, test_fn.__name__, False, duration, str(e))
                print(f"Test {idx}/10: {test_fn.__name__} -> CRASHED ({duration:.2f}s)")
        
        self.print_report()

    def test_sanity_check(self):
        messages = [{"role": "user", "content": "Ping"}]
        res = self.client.generate_content(messages)
        if res and len(res.strip()) > 0:
            return True, f"Received: {res.strip()[:30]}..."
        return False, "Empty response received."

    def test_system_prompt_adherence(self):
        messages = [
            {"role": "system", "content": "You are an absolute minimalist. You only reply with the word 'ACK'."},
            {"role": "user", "content": "Can you explain backpropagation in deep neural networks?"}
        ]
        res = self.client.generate_content(messages).strip()
        if res == "ACK" or "ACK" in res:
            return True, f"System prompt successfully forced constraint. Output: {res}"
        return False, f"Model failed strict system prompt role constraint. Output: {res}"

    def test_deterministic_temperature(self):
        messages = [{"role": "user", "content": "Generate a unique random 5-digit number sequence."}]
        res1 = self.client.generate_content(messages, temperature=0.0).strip()
        res2 = self.client.generate_content(messages, temperature=0.0).strip()
        if res1 == res2:
            return True, f"Deterministic enforcement successful. Response 1 matches Response 2: {res1}"
        return False, f"Non-deterministic execution at 0.0 temperature. Diff: {res1} vs {res2}"

    def test_high_temperature_variance(self):
        messages = [{"role": "user", "content": "Write a weird creative metaphor for low-level pointer arithmetic."}]
        res1 = self.client.generate_content(messages, temperature=1.0)
        res2 = self.client.generate_content(messages, temperature=1.0)
        if res1 != res2:
            return True, "High temperature test succeeded. Outputs diverged organically as expected."
        return False, "Outputs matched identically despite high temperature setting."

    def test_json_structure_generation(self):
        messages = [
            {"role": "system", "content": "Return a raw JSON object containing the keys 'status' and 'code'. Do not include markdown formatting or backticks."},
            {"role": "user", "content": "Output status success and code 200."}
        ]
        res = self.client.generate_content(messages)
        try:
            parsed = json.loads(res.strip())
            if "status" in parsed and "code" in parsed:
                return True, "Valid structured JSON extracted perfectly."
            return False, f"Missing required keys in valid JSON block. Structure: {parsed}"
        except json.JSONDecodeError:
            return False, f"Failed to parse raw output into JSON. Content: {res}"

    def test_large_payload_handling(self):
        synthetic_context = "Machine Learning infrastructure scaling requires robust distributed systems architecture. " * 300
        messages = [
            {"role": "system", "content": f"Analyze this context: {synthetic_context}"},
            {"role": "user", "content": "Summarize the primary thesis of the context block in three words."}
        ]
        res = self.client.generate_content(messages)
        if res and len(res) > 0:
            return True, f"Successfully digested bloated context payload. Response: {res.strip()}"
        return False, "Failed to resolve response on extended input sequence."

    def test_unicode_and_math_symbols(self):
        messages = [{"role": "user", "content": "Express the standard loss optimization function for a support vector machine using correct LaTeX equations."}]
        res = self.client.generate_content(messages)
        if "$" in res or "\\" in res:
            return True, "Successfully rendered raw mathematical syntax and unicode characters."
        return False, f"Failed to output symbolic math notation. Response: {res}"

    def test_strict_token_limitation(self):
        messages = [{"role": "user", "content": "Write an exceptionally long essay detailing the entire history of the Linux kernel."}]
        res = self.client.generate_content(messages)
        if len(res) > 0:
            return True, f"Tokens handled up to system context thresholds without unexpected API aborts. Word count: {len(res.split())}"
        return False, "Null response generated during long form generation."

    def test_consecutive_rapid_fire(self):
        messages = [{"role": "user", "content": "Quick trace check."}]
        for i in range(3):
            self.client.generate_content(messages)
        return True, "Successfully executed 3 consecutive network round-trips without triggering rate limits."

    def test_malformed_payload_resilience(self):
        try:
            self.client.generate_content([])
            return False, "API accepted a completely empty messages array without throwing an exception."
        except Exception:
            return True, "Gracefully threw error when client received malformed/empty payload arrays."

    def print_report(self):
        print("\n" + "="*70)
        print("                           FINAL TEST REPORT")
        print("="*70)
        passed_count = sum(1 for t in self.results if t["success"])
        for t in self.results:
            status = "[PASS]" if t["success"] else "[FAIL]"
            print(f"{status} Test {t['id']}: {t['name']} ({t['time']})")
            if not t["success"]:
                print(f"       -> Details: {t['details']}")
        print("="*70)
        print(f"TOTAL: {passed_count}/{len(self.results)} Passed.")
        print("="*70)

if __name__ == "__main__":
    suite = GroqTestSuite()
    suite.run_all()