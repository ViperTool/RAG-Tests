import { useEffect, useRef } from "react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { siteConfig } from "@/config/site"
import Message from "./Message"
import ReactMarkdown from "react-markdown"

export default function ChatBox({ messages, isLoading }) {
  const messagesEndRef = useRef(null)
  const scrollAreaRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  return (
    <div className="flex-1 overflow-hidden">
      <ScrollArea className="h-full" ref={scrollAreaRef}>
        <div className="space-y-2 p-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-center space-y-6 animate-fade-in">
              <div className="flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-br from-blue-500/20 to-blue-700/20 border border-blue-500/30">
                <span className="text-4xl">💡</span>
              </div>
              <div className="space-y-2 max-w-md">
                <h3 className="text-lg font-semibold text-foreground">
                  Готов помочь!
                </h3>
                <div className="text-sm text-muted-foreground prose prose-invert">
                  <ReactMarkdown>{siteConfig.chat.welcomeMessage}</ReactMarkdown>
                </div>
              </div>
            </div>
          ) : (
            <>
              {messages.map((msg, idx) => (
                <Message
                  key={idx}
                  role={msg.role}
                  content={msg.content}
                  isLoading={msg.isLoading}
                />
              ))}
              {isLoading && <Message role="assistant" content="" isLoading={true} />}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>
      </ScrollArea>
    </div>
  )
}