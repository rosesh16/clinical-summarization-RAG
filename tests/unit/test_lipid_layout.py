import unittest

from chasm_rag.evidence.measurement_parser import parse_lipid_profile_layout
from chasm_rag.evidence.measurement_validator import classify_measurement


class LipidLayoutTests(unittest.TestCase):
    def test_column_major_lipid_layout_reconstructs_only_lab_rows(self) -> None:
        text = """Registered on: 02:31 PM 02 Dec, 2X
Age : 21 Years
LIPID PROFILE
Investigation Result Reference Value Unit
mg/dL
< 200.00
High
250.00
Cholesterol Total
Spectrophotometry
Triglycerides
mg/dL
< 150.00
100.00
Spectrophotometry
HDL Cholesterol
mg/dL
> 40.00
50.00
Spectrophotometry
LDL Cholesterol
mg/dL
< 100.00
High
190.00
Calculated
mg/dL
< 30.00
VLDL Cholesterol
10.00
Calculated
mg/dL
< 130.00
Non-HDL Cholesterol
100.00
Calculated
Smart Pathology Laboratory"""
        measurements = parse_lipid_profile_layout(text, 1)
        rows = {item.test_name: item for item in measurements}
        self.assertEqual(set(rows), {"Cholesterol Total", "Triglycerides", "HDL Cholesterol", "LDL Cholesterol", "VLDL Cholesterol", "Non-HDL Cholesterol"})
        self.assertEqual((rows["Cholesterol Total"].value, rows["Cholesterol Total"].reference_range.raw), (250, "< 200.00"))
        self.assertEqual((rows["LDL Cholesterol"].value, rows["LDL Cholesterol"].reference_range.raw), (190, "< 100.00"))
        self.assertEqual(rows["VLDL Cholesterol"].reference_range.raw, "< 30.00")
        self.assertEqual(classify_measurement(rows["VLDL Cholesterol"]), "WITHIN SUPPLIED RANGE")
