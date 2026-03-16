# Bearing Friction Torque Web

这是一个与旧站完全分开的独立项目，专门用于球轴承摩擦力矩在线计算。网页会根据载荷、转速、几何和润滑参数，计算 EHL 膜厚、`kappa`、`lambda`、逐钢球摩擦力矩和总摩擦力矩。

## Files

- `friction_torque_app.py`: 单文件 Flask 应用，包含计算模型、页面模板、输入校验、CSV 导出和健康检查
- `requirements.txt`: Python 依赖
- `render.yaml`: Render Blueprint 配置
- `README.md`: 使用说明

## Local Run

```bash
pip install -r requirements.txt
python3 friction_torque_app.py
```

浏览器打开 `http://127.0.0.1:5001`。

## GitHub

```bash
git init -b main
git remote add origin https://github.com/yeluofengqiao/bearing-friction-torque-web.git
git add .
git commit -m "Initial commit for bearing friction torque web app"
git push -u origin main
```

## Render

1. 登录 [Render](https://render.com/)
2. 选择 `New +` -> `Blueprint`
3. 连接 GitHub 仓库 `yeluofengqiao/bearing-friction-torque-web`
4. Render 会自动读取本仓库中的 `render.yaml`
5. 确认后创建服务

部署成功后会得到一个新的独立网页链接，与旧站完全分开。
