import pytest
import os

if __name__ == "__main__":
    pytest.main([
        "-v",
        "--tb=short",
        "--disable-warnings",
        "-m", "not performance",
        "--alluredir=reports/allure-results",
        "--clean-alluredir"
    ])

    # Generate and open report automatically
    os.system("allure serve reports/allure-results")