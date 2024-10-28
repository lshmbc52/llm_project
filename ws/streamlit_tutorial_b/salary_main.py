# import streamlit as st

# from package.salary import KoreanSalaryCalculator

# st.sidebar.title("About")
# st.sidebar.info("연봉계산기")

# calculator = KoreanSalaryCalculator()

# st.title("급여 명세서")

# annual_salary = st.number_input("연봉 입력", step= 1000000)
# bonus_ratio = st.number_input("보너스 비율")


# result = calculator.calculate_monthly_salary(annual_salary, bonus_ratio)

# st.writ(f'총지급액: {result["gross_salary"]:, .0f}원')
# st.write(f'\n공제내역')
# st.write(f"국민연금: {result['insurance_premiums']['national_pension']:,.0f}원")
# st.write(f"건강보험: {result['insurance_premiums']['health_insurance']:,.0f}원")
# st.write(f"장기요양보험: {result['insurance_premiums']['long_term_care']:,.0f}원")
# st.write(f"고용보험: {result['insurance_premiums']['employment_insurance']:,.0f}원")
# st.write(f"소득세: {result['income_tax']:,.0f}원")
# st.write(f"\n총 공제액: {result['deductions']:,.0f}원")
# st.write(f"실수령액: {result['net_salary']:,.0f}원")

import streamlit as st
from package.salary import KoreanSalaryCalculator

st.sidebar.title("About")
st.sidebar.info("연봉 계산기")


# 사용 예시
calculator = KoreanSalaryCalculator()

st.title("=== 급여 명세서 ===")

# 연봉 5000만원, 상여금 비율 20%인 경우의 월급 계산
annual_salary = st.number_input("연봉 입력",step=1000000)
bonus_ratio = st.number_input("보너스 비율")

result = calculator.calculate_monthly_salary(annual_salary, bonus_ratio)
st.write(f"총 지급액: {result['gross_salary']:,.0f}원")
st.write(f"\n공제 내역:")
st.write(f"국민연금: {result['insurance_premiums']['national_pension']:,.0f}원")
st.write(f"건강보험: {result['insurance_premiums']['health_insurance']:,.0f}원")
st.write(f"장기요양보험: {result['insurance_premiums']['long_term_care']:,.0f}원")
st.write(f"고용보험: {result['insurance_premiums']['employment_insurance']:,.0f}원")
st.write(f"소득세: {result['income_tax']:,.0f}원")
st.write(f"\n총 공제액: {result['deductions']:,.0f}원")
st.write(f"실수령액: {result['net_salary']:,.0f}원")