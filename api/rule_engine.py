import json
import logging

logger = logging.getLogger(__name__)

class RuleEngine:
    def __init__(self, rules_file_path):
        with open(rules_file_path, 'r') as f:
            self.data = json.load(f)
        self.rules = sorted(self.data.get('rules', []), key=lambda x: x.get('priority', 0), reverse=True)
        self.process_name = self.data.get('process')

    def evaluate(self, context: dict):
        """
        Evaluates the context against the loaded rules.
        Returns the 'then' block of the first matching rule (highest priority).
        """
        logger.info(f"Evaluating rules for {self.process_name} with context: {context}")
        
        for rule in self.rules:
            if self._matches(rule.get('when', {}), context):
                logger.info(f"Rule matched: {rule.get('id')}")
                return rule.get('then')
        
        return None

    def _matches(self, conditions, context):
        for field, condition in conditions.items():
            value = context.get(field)
            
            # If value is missing in context, we assume it doesn't match if a condition is required
            if value is None:
                return False

            if isinstance(condition, dict):
                # Handle operators like min, max
                val_num = self._to_number(value)
                
                if 'min' in condition:
                    if val_num is None or not (val_num >= condition['min']):
                        return False
                if 'max' in condition:
                    if val_num is None or not (val_num <= condition['max']):
                        return False
            else:
                # Exact match
                if value != condition:
                    return False
        return True

    def _to_number(self, val):
        """Attempts to convert a value to a float for comparison."""
        if isinstance(val, (int, float)):
            return val
        if isinstance(val, str):
            import re
            # Try to find a number in the string (e.g. "3 botellas" -> 3.0)
            # This is simple; won't handle "tres" but handles mixed strings
            match = re.search(r'(-?\d+(\.\d+)?)', val)
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass
        return None

    def get_missing_info(self, context):
        """
        Checks which required fields are missing from the context.
        """
        required = self.data.get('required_fields', [])
        missing = [field for field in required if field not in context or context[field] is None or context[field] == ""]
        
        if missing:
            behavior = self.data.get('missing_info_behavior', {})
            question_def = behavior.get('questions', {}).get(missing[0])
            return {
                "status": "NEED_INFO",
                "missing_field": missing[0],
                "question": question_def.get('question') if question_def else f"Please provide {missing[0]}",
                "options": question_def.get('options') if question_def else None
            }
        return None
