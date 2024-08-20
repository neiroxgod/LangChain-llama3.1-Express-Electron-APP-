import readline from "readline";
import axios from "axios";

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const API_URL = "http://localhost:3000/api/generate";

const main = () => {
  rl.question("Введите ваш запрос: ", async (prompt) => {
    try {
      const response = await axios.post(API_URL, { prompt });
      console.log("Ответ:", response.data.response);
    } catch (error) {
      console.error(
        "Ошибка:",
        error.response ? error.response.data : error.message
      );
    }
    main();
  });
};

main();
