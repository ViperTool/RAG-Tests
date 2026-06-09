export const siteConfig = {
  title: "RAG Knowledge Base",
  shortTitle: "RAG Assistant",
  version: "2.0",
  description: "Интеллектуальный помощник на основе ваших данных",

  chat: {
    placeholder: "Задайте ваш вопрос...",
    welcomeMessage: `Я — интеллектуальный ассистент, готовый ответить на ваши вопросы.

**Что я умею:**
- 📚 Отвечать на вопросы по документам
- 🔍 Искать релевантную информацию
- 🎯 Давать точные ответы

*Начните с примеров ниже или задайте свой вопрос.*`
  },

  examples: [
    "Как начать работу с системой?",
    "Какие документы доступны?",
    "Как настроить поиск?",
    "Где найти документацию?"
  ],

  links: {
    documentation: "#",
    support: "#"
  },

  ui: {
    showExamples: true,
    maxExamplesToShow: 4
  }
}