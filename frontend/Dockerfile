FROM node:22-alpine

WORKDIR /app

COPY package.json ./
RUN npm install --legacy-peer-deps

COPY . .

EXPOSE 5173

CMD ["npx", "vite", "--host", "0.0.0.0"]
