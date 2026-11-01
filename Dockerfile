FROM python:3.12-slim

WORKDIR /app

# 先只拷依赖清单，这样改代码不会让依赖层缓存失效（每次构建不用重装）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 再拷代码
COPY . .

EXPOSE 8000

# 容器里必须监听 0.0.0.0，否则宿主机访问不到
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
