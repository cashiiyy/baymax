import math
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class ToolResult(BaseModel):
    tool_name: str
    status: str = "success"
    result: Dict[str, Any]
    summary: str

class MedicalTool:
    """Base interface for all medical tools."""
    name: str = "base_tool"
    description: str = ""

    def execute(self, **kwargs) -> ToolResult:
        raise NotImplementedError

class BMICalculatorTool(MedicalTool):
    name = "bmi_calculator"
    description = "Calculates Body Mass Index (BMI) given weight (kg) and height (cm or m)."

    def execute(self, weight_kg: float, height_cm: float) -> ToolResult:
        height_m = height_cm / 100.0 if height_cm > 3.0 else height_cm
        if height_m <= 0:
            return ToolResult(tool_name=self.name, status="error", result={}, summary="Height must be greater than zero.")
        
        bmi = round(weight_kg / (height_m ** 2), 1)
        category = "Normal"
        if bmi < 18.5:
            category = "Underweight"
        elif bmi >= 25.0 and bmi < 30.0:
            category = "Overweight"
        elif bmi >= 30.0:
            category = "Obesity"

        return ToolResult(
            tool_name=self.name,
            result={"bmi": bmi, "category": category, "weight_kg": weight_kg, "height_m": height_m},
            summary=f"BMI: {bmi} ({category})"
        )

class BSACalculatorTool(MedicalTool):
    name = "bsa_calculator"
    description = "Calculates Body Surface Area (BSA) in m² using Mosteller formula."

    def execute(self, weight_kg: float, height_cm: float) -> ToolResult:
        bsa = round(math.sqrt((weight_kg * height_cm) / 3600.0), 2)
        return ToolResult(
            tool_name=self.name,
            result={"bsa_m2": bsa, "formula": "Mosteller"},
            summary=f"Calculated BSA: {bsa} m²"
        )

class UnitConverterTool(MedicalTool):
    name = "unit_converter"
    description = "Converts clinical units (e.g. Temperature °F to °C, Glucose mg/dL to mmol/L)."

    def execute(self, value: float, from_unit: str, to_unit: str) -> ToolResult:
        from_u = from_unit.lower()
        to_u = to_unit.lower()
        converted = value

        if "f" in from_u and "c" in to_u:
            converted = round((value - 32) * 5 / 9, 1)
        elif "c" in from_u and "f" in to_u:
            converted = round((value * 9 / 5) + 32, 1)
        elif "mg/dl" in from_u and "mmol/l" in to_u:
            converted = round(value / 18.0, 2)
        elif "mmol/l" in from_u and "mg/dl" in to_u:
            converted = round(value * 18.0, 1)

        return ToolResult(
            tool_name=self.name,
            result={"original": f"{value} {from_unit}", "converted": f"{converted} {to_unit}"},
            summary=f"{value} {from_unit} = {converted} {to_unit}"
        )

class MedicalAbbreviationTool(MedicalTool):
    name = "medical_abbreviations"
    description = "Look up common medical abbreviations."

    ABBREVIATIONS = {
        "cpr": "Cardiopulmonary Resuscitation",
        "bp": "Blood Pressure",
        "hr": "Heart Rate",
        "bmi": "Body Mass Index",
        "stat": "Immediately / At once",
        "prn": "As needed (pro re nata)",
        "bid": "Twice a day (bis in die)",
        "tid": "Three times a day (ter in die)"
    }

    def execute(self, abbr: str) -> ToolResult:
        clean = abbr.lower().strip()
        meaning = self.ABBREVIATIONS.get(clean, "Abbreviation not found in reference dictionary.")
        return ToolResult(
            tool_name=self.name,
            result={"abbreviation": abbr, "meaning": meaning},
            summary=f"{abbr.upper()}: {meaning}"
        )

class MedicalToolRegistry:
    """Registry maintaining available plugin tools."""

    def __init__(self):
        self.tools: Dict[str, MedicalTool] = {
            "bmi_calculator": BMICalculatorTool(),
            "bsa_calculator": BSACalculatorTool(),
            "unit_converter": UnitConverterTool(),
            "medical_abbreviations": MedicalAbbreviationTool()
        }

    def get_tool(self, name: str) -> Optional[MedicalTool]:
        return self.tools.get(name)

    def execute_tool(self, name: str, **kwargs) -> ToolResult:
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(tool_name=name, status="error", result={}, summary=f"Tool {name} not registered.")
        return tool.execute(**kwargs)

tool_registry = MedicalToolRegistry()
