const { contextBridge } = require("electron");
const axios = require("axios");

contextBridge.exposeInMainWorld("myAPI", {
  sendHttpRequest: async (url, method, body) => {
    try {
      const response = await axios({
        url: url,
        method: method,
        data: body,
      });
      if (response.status === 200) {
        return response.data;
      } else {
        throw new Error("Network response was not ok.");
      }
    } catch (error) {
      throw error;
    }
  },
});
