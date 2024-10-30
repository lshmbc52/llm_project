import streamlit as st
import pandas as pd

person = pd.DataFrame({"work": ["baby_seater","environment_watcher","nursing_care_worker"],
                       "region":["서울","부산","대구"],
                       "experience":[10,2,1]})


class JobSearch:
    def __init__(self,person):
        self.job = person

    def search(self,keyword,location,experience):

        return self.person[(self.person['work'].str.contains(keyword)) &
                           (self.person['region']== location) &
                           (self.person['experience'] >= experience)]

def main():
    job_search = JobSearch(person)
    st.title("Senior job search")

    keyword = st.text_input("담당했던 업무")
    location = st.selectbox("지역", person["region"])
    experience = st.slider( "최소경력", 0, 10 )

    if st.button("검색"):
        results = job_search.search(keyword,location, experience)
        st.dataframe(results)


if __name__== "__main__":
    main()






        