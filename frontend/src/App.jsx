import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import Header from "@/components/Header"
import HeroBanner from "@/components/HeroBanner"
import ChatBox from "@/components/ChatBox"
import InputArea from "@/components/InputArea"
import Examples from "@/components/Examples"
import Footer from "@/components/Footer"
import { queryRag } from "@/api/client"
import { siteConfig } from "@/config/site"
import { Trash2 } from "lucide-react"

function App() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSend = async (text) => {
    setError(null)

    // Add user message
    setMessages(prev => [...prev, { role: "user", content: text }])
    setIsLoading(true)

    try {
      const response = await queryRag(text)
      setMessages(prev => [...prev, {
        role: "assistant",
        content: response.answer
      }])
    } catch (err) {
      setError(err.message)
      setMessages(prev => [...prev, {
        role: "assistant",
        content: `⚠️ Произошла ошибка: ${err.message}`
      }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleClear = () => {
    setMessages([])
    setError(null)
  }

  return (
    <div className="relative flex min-h-screen flex-col bg-background">
      {/* Gradient background effects */}
      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/5 rounded-full blur-3xl" />
      </div>

      <Header />
      <HeroBanner />

      <main className="flex-1 flex flex-col">
        <Card className="container max-w-4xl mx-auto mt-6 mb-6 border-border/40 shadow-2xl shadow-blue-500/5 overflow-hidden flex flex-col" style={{ minHeight: "600px", maxHeight: "80vh" }}>
          {error && (
            <div className="bg-destructive/10 border-b border-destructive/20 px-4 py-3 text-sm text-destructive flex items-center justify-between">
              <span>{error}</span>
              <Button variant="ghost" size="sm" onClick={() => setError(null)} className="h-6 w-6 p-0">
                ×
              </Button>
            </div>
          )}

          <ChatBox messages={messages} isLoading={isLoading} />

          {messages.length === 0 && (
            <Examples onSelect={handleSend} disabled={isLoading} />
          )}

          <InputArea onSend={handleSend} disabled={isLoading} />
        </Card>

        {messages.length > 0 && (
          <div className="container max-w-4xl mx-auto mb-6 flex justify-center">
            <Button
              variant="outline"
              size="sm"
              onClick={handleClear}
              disabled={isLoading}
              className="rounded-full gap-2 hover:bg-destructive/10 hover:text-destructive hover:border-destructive/30 transition-all"
            >
              <Trash2 className="h-4 w-4" />
              Очистить чат
            </Button>
          </div>
        )}
      </main>

      <Footer />
    </div>
  )
}

export default App