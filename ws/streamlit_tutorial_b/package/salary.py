class KoreanSalaryCalculator:
    def __init__(self):
        self.tax_brackets = [
            (12000000, 0.06),  # 1200만원 이하
            (46000000, 0.15),  # 4600만원 이하
            (88000000, 0.24),  # 8800만원 이하
            (150000000, 0.35), # 1.5억원 이하
            (300000000, 0.38), # 3억원 이하
            (500000000, 0.40), # 5억원 이하
            (1000000000, 0.42), # 10억원 이하
            (float('inf'), 0.45) # 10억원 초과
        ]

        self.national_pension_rate = 0.045
        self.health_insurance_rate = 0.0349
        self.employment_insurance_rate = 0.009
        self.long_term_care_rate = 0.1227

    def calculate_income_tax(self,annual_salary):
        tax = 0
        remaining_salary = annual_salary
        prev_bracket = 0

        for bracket, rate in self.tax_brackets:

            if remaining_salary <= 0:
                break

            taxable_in_bracket = min(remaining_salary, bracket - prev_bracket)
            tax += taxable_in_bracket * rate
            remaining_salary -= taxable_in_bracket
            prev_bracket = bracket

        return tax

    def calculate_insurance_premiums(self,monthly_salary):
        national_pension = monthly_salary * self.national_pension_rate
        health_insurance = monthly_salary * self.health_insurance_rate
        long_term_care = health_insurance * self.long_term_care_rate
        employment_insurance = monthly_salary *self.employment_insurance_rate

        return {
            "national_pension": national_pension,
            "health_insurance":health_insurance,
            "long_term_care": long_term_care,
            "employment_insurance": employment_insurance,
            "total": national_pension + health_insurance + long_term_care + employment_insurance
        }

    def calculate_monthly_salary(self, annual_salary, bonus_ratio= 0):
        bonus = annual_salary * bonus_ratio
        base_monthly_salary = (annual_salary - bonus) / 12
        monthly_bonus = bonus / 12

        monthly_salary = base_monthly_salary + monthly_bonus

        insurance = self.calculate_insurance_premiums(monthly_salary)

        monthly_income_tax = self.calculate_income_tax(annual_salary) / 12

        net_salary = monthly_salary - insurance ["total"] - monthly_income_tax

        return {
            "gross_salary": monthly_salary,
            "insurance_premiums": insurance,
            "income_tax": monthly_income_tax,
            "net_salary":net_salary,
            "deductions":insurance["total"] + monthly_income_tax
        }

