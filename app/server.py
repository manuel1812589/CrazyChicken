from app.app import app

if __name__ == "__main__":
    app.run(
        debug=False,
        dev_tools_hot_reload=False,
        dev_tools_ui=False,
        dev_tools_props_check=False,
    )
