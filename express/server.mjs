import express from "express";
import { RetrievalQAChain, loadQAStuffChain } from "langchain/chains";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { OllamaEmbeddings, Ollama } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { PromptTemplate } from "@langchain/core/prompts";
const app = express();
app.use(express.json()); // Разрешаем JSON-запросы

const urls = [
  "https://bigbencrm.freshdesk.com/support/solutions/articles/70000139889-%D0%BA%D0%B0%D0%BA-%D1%81%D0%BE%D0%B7%D0%B4%D0%B0%D1%82%D1%8C-%D0%B3%D1%80%D1%83%D0%BF%D0%BF%D1%83",
  "https://bigbencrm.freshdesk.com/support/solutions/articles/70000137575-%D0%BA%D0%B0%D0%BA-%D1%83%D0%B2%D0%B8%D0%B4%D0%B5%D1%82%D1%8C-%D1%81%D0%BF%D0%B8%D1%81%D0%BE%D0%BA-%D0%B4%D0%BE%D0%BB%D0%B6%D0%BD%D0%B8%D0%BA%D0%BE%D0%B2-%D0%BF%D0%BE-%D0%B3%D1%80%D1%83%D0%BF%D0%BF%D0%B0%D0%BC",
  "https://bigbencrm.freshdesk.com/support/solutions/articles/70000140319-%D0%BA%D0%B0%D0%BA-%D0%BE%D1%82%D1%84%D0%B8%D0%BB%D1%8C%D1%82%D1%80%D0%BE%D0%B2%D0%B0%D1%82%D1%8C-%D0%B3%D1%80%D1%83%D0%BF%D0%BF%D1%8B",
  "https://bigbencrm.freshdesk.com/support/solutions/articles/70000140435-%D0%BA%D0%B0%D0%BA-%D0%BD%D0%B0%D1%81%D1%82%D1%80%D0%BE%D0%B8%D1%82%D1%8C-%D1%80%D0%B0%D1%81%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5-%D0%BF%D0%BE-%D0%BD%D0%B5%D1%81%D0%BA%D0%BE%D0%BB%D1%8C%D0%BA%D0%B8%D0%BC-%D0%BF%D1%80%D0%B5%D0%B4%D0%BC%D0%B5%D1%82%D0%B0%D0%BC-%D0%B0%D1%83%D0%B4%D0%B8%D1%82%D0%BE%D1%80%D0%B8%D1%8F%D0%BC-%D0%B8%D0%BB%D0%B8-%D0%BF%D1%80%D0%B5%D0%BF%D0%BE%D0%B4%D0%B0%D0%B2%D0%B0%D1%82%D0%B5%D0%BB%D1%8F%D0%BC-%D0%B2-%D0%BE%D0%B4%D0%BD%D0%BE%D0%B9-%D0%B3%D1%80%D1%83%D0%BF%D0%BF%D0%B5",
  "https://bigbencrm.freshdesk.com/support/solutions/articles/70000154076-%D0%BA%D0%B0%D0%BA-%D1%83%D0%B2%D0%B8%D0%B4%D0%B5%D1%82%D1%8C-%D1%81%D0%BF%D0%B8%D1%81%D0%BE%D0%BA-%D0%B3%D1%80%D1%83%D0%BF%D0%BF-%D0%BF%D0%BE-%D0%BE%D1%82%D0%B4%D0%B5%D0%BB%D1%8C%D0%BD%D0%BE%D0%BC%D1%83-%D1%84%D0%B8%D0%BB%D0%B8%D0%B0%D0%BB%D1%83-%D0%B8%D0%BB%D0%B8-%D0%B0%D1%83%D0%B4%D0%B8%D1%82%D0%BE%D1%80%D0%B8%D0%B8",
  "https://bigbencrm.freshdesk.com/support/solutions/articles/70000154078-%D0%BA%D0%B0%D0%BA-%D0%B4%D0%BE%D0%B1%D0%B0%D0%B2%D0%B8%D1%82%D1%8C-%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%BA%D0%BE%D0%B2-%D0%B2-%D0%B3%D1%80%D1%83%D0%BF%D0%BF%D1%83",
  "https://bigbencrm.freshdesk.com/support/solutions/articles/70000154080-%D0%BA%D0%B0%D0%BA-%D0%B7%D0%B0%D0%BC%D0%B5%D0%BD%D0%B8%D1%82%D1%8C-%D0%BF%D1%80%D0%B5%D0%BF%D0%BE%D0%B4%D0%B0%D0%B2%D0%B0%D1%82%D0%B5%D0%BB%D1%8F-%D0%B2-%D0%B3%D1%80%D1%83%D0%BF%D0%BF%D0%B5-%D0%BD%D0%B0-%D0%BE%D0%B4%D0%B8%D0%BD-%D0%B8%D0%BB%D0%B8-%D0%BD%D0%B5%D1%81%D0%BA%D0%BE%D0%BB%D1%8C%D0%BA%D0%BE-%D1%83%D1%80%D0%BE%D0%BA%D0%BE%D0%B2",
  "https://bigbencrm.freshdesk.com/support/solutions/articles/70000154083-%D0%BA%D0%B0%D0%BA-%D0%BF%D0%B5%D1%80%D0%B5%D0%BD%D0%B5%D1%81%D1%82%D0%B8-%D0%B3%D1%80%D1%83%D0%BF%D0%BF%D1%83-%D0%B2-%D0%B4%D1%80%D1%83%D0%B3%D1%83%D1%8E-%D0%B0%D1%83%D0%B4%D0%B8%D1%82%D0%BE%D1%80%D0%B8%D1%8E",
  "https://bigbencrm.freshdesk.com/support/solutions/articles/70000154084-%D1%82%D0%B8%D0%BF%D1%8B-%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D1%8F",
  "https://bigbencrm.freshdesk.com/support/solutions/articles/70000154087-%D0%BA%D0%B0%D0%BA-%D0%BE%D1%82%D0%BC%D0%B5%D1%87%D0%B0%D1%82%D1%8C-%D0%BF%D0%BE%D1%81%D0%B5%D1%89%D0%B0%D0%B5%D0%BC%D0%BE%D1%81%D1%82%D1%8C",
];

const allDocs = [];

for (const url of urls) {
  const loaderWithSelector = new CheerioWebBaseLoader(url, {
    selector: ".c-wrapper",
  });
  const rawDocs = await loaderWithSelector.load();
  allDocs.push(...rawDocs);
}

const ollamaEmbeddings = new OllamaEmbeddings({
  model: "llama3.1",
});

//Create a text splitter
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

const docs = await splitter.splitDocuments(allDocs);

const vectorstore = await MemoryVectorStore.fromDocuments(
  docs,
  ollamaEmbeddings
);
const retriever = vectorstore.asRetriever();

const llm = new Ollama({
  model: "llama3.1",
});

const template = `Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always say "Спасибо за вопрос!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:`;

const QA_CHAIN_PROMPT = new PromptTemplate({
  inputVariables: ["context", "question"],
  template,
});

const chain = new RetrievalQAChain({
  combineDocumentsChain: loadQAStuffChain(llm, { prompt: QA_CHAIN_PROMPT }),
  retriever,
  returnSourceDocuments: true,
  inputKey: "question",
});

app.post("/api/generate", async (req, res) => {
  const prompt = req.body.prompt;

  const response = await chain.call({
    question: prompt,
  });

  const responseText = response.text;
  console.log(response.text);

  res.json({ response: responseText });
});

app.listen(3000, () => {
  console.log("Сервер запущен на порту 3000");
});
