import streamlit as st
from webui import main

# 用户名和密码的默认值
USER_CREDENTIALS = {
    "admin": ("admin", "123456"),  # 管理员用户名和密码
    "user": ("zwk", "123")   # 普通用户用户名和密码
}

# 检查会话状态中是否有登录状态，如果没有，初始化为 False
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'admin' not in st.session_state:
    st.session_state.admin = False

def login_page():
    with st.form("login_form"):
        st.title("登录")
        username = st.text_input("用户名", value="")
        password = st.text_input("密码", value="", type="password")
        submit = st.form_submit_button("登录")

        if submit:
            # 检查用户名和密码是否与管理员或普通用户匹配
            if (username, password) == USER_CREDENTIALS["admin"]:
                st.success("管理员登录成功！")
                st.session_state.logged_in = True
                st.session_state.admin = True
                st.experimental_rerun()  # 重新运行脚本以显示主页面
            elif (username, password) == USER_CREDENTIALS["user"]:
                st.success("用户登录成功！")
                st.session_state.logged_in = True
                st.session_state.admin = False
                st.experimental_rerun()
            else:
                st.error("用户名或密码错误，请重新输入。")

if __name__ == "__main__":
    # 如果用户已登录，则显示应用的其他部分
    if st.session_state.logged_in:
        main(st.session_state.admin)
    else:
        # 如果用户未登录，则显示登录页面
        login_page()
