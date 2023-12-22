import streamlit as st 
from quality_evaluation import *


def get_photo():
    return st.file_uploader("Додайте фото етикети вина", type=["jpg", "png", "heic"])    

def main():
    st.title("Визначення якості вина")
    uploaded_photo = get_photo()
    get_quality = st.button("Визначити якість")

    if get_quality:
        file_bytes = np.asarray(bytearray(uploaded_photo.read()), dtype=np.uint8)
        im = cv2.imdecode(file_bytes, 1)
        score = evaluate_quality(im)

        st.image(uploaded_photo, caption="Ваше завантажене фото", width=420)
        st.subheader("Оцінка вина: " + str(score) + "/10.0")
        

if __name__ == "__main__":
    main()

