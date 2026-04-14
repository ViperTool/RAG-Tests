import ReactMarkdown from "react-markdown"
import { ScrollArea } from "@/components/ui/scroll-area"
import { siteConfig } from "@/config/site"
import { cn } from "@/lib/utils"

export default function Message({ role, content, isLoading }) {
  if (isLoading) {
    return (
      <div className="flex gap-4 p-4 animate-fade-in">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-blue-700">
          <span className="text-lg">🤖</span>
        </div>
        <div className="flex-1 space-y-2">
          <div className="flex gap-1">
            <span className="h-2 w-2 rounded-full bg-blue-500 animate-bounce [animation-delay:-0.3s]" />
            <span className="h-2 w-2 rounded-full bg-blue-500 animate-bounce [animation-delay:-0.15s]" />
            <span className="h-2 w-2 rounded-full bg-blue-500 animate-bounce" />
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={cn(
      "flex gap-4 p-4 animate-fade-in",
      role === "user" && "flex-row-reverse"
    )}>
      <div className={cn(
        "flex h-10 w-10 shrink-0 items-center justify-center rounded-full shadow-lg",
        role === "user"
          ? "bg-gradient-to-br from-purple-500 to-purple-700"
          : "bg-gradient-to-br from-blue-500 to-blue-700"
      )}>
        <span className="text-lg">{role === "user" ? "👤" : "🤖"}</span>
      </div>

      <div className={cn(
        "flex-1 rounded-2xl border p-4 shadow-sm",
        role === "user"
          ? "bg-gradient-to-br from-purple-600/20 to-purple-800/20 border-purple-500/20"
          : "bg-gradient-to-br from-blue-600/10 to-slate-800/50 border-blue-500/20"
      )}>
        <ScrollArea className="max-h-[600px] pr-4">
          <div className="prose prose-invert prose-sm max-w-none">
            <ReactMarkdown>{content}</ReactMarkdown>
          </div>
        </ScrollArea>
      </div>
    </div>
  )
}