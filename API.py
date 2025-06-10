from flask import Flask, request, jsonify

app = Flask(__name__)

# 创建一个路由，用于处理问候请求
@app.route('/greet', methods=['POST'])
def greet():
    data = request.json  # 获取请求的JSON数据
    name = data.get('name')  # 从数据中提取名字
    message = f"Hello, {name}!"  # 调用函数生成消息
    return jsonify({"message": message})  # 返回JSON格式的响应

if __name__ == '__main__':
    app.run(debug=True)  # 运行Flask应用