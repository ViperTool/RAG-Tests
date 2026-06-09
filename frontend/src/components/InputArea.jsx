import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { siteConfig } from "@/config/site"
import { Send, Loader2 } from "lucide-react"

export default function InputArea({ onSend, disabled }) {
  const [input, setInput] = useState("")
  const textareaRef = useRef(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim() || disabled) return

    onSend(input.trim())
    setInput("")
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const handleInput = (e) => {
    setInput(e.target.value)
    const textarea = e.target
    textarea.style.height = "auto"
    textarea.style.height = Math.min(textarea.scrollHeight, 150) + "px"
  }

  return (
    <div className="border-t border-border/40 bg-background/50 backdrop-blur-sm p-4">
      <form onSubmit={handleSubmit} className="container max-w-4xl mx-auto space-y-2">
        <div className="relative flex gap-2">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder={siteConfig.chat.placeholder}
            disabled={disabled}
            rows={1}
            className="flex-1 min-h-[48px] max-h-[150px] resize-none rounded-xl border border-input bg-background/80 px-4 py-3 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
          />
          <Button
            type="submit"
            disabled={!input.trim() || disabled}
            className="shrink-0 rounded-xl px-6"
          >
            {disabled ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <Send className="h-5 w-5" />
            )}
          </Button>
        </div>
        <p className="text-xs text-center text-muted-foreground">
          Нажмите Enter для отправки, Shift+Enter для новой строки
        </p>
      </form>
    </div>
  )
}