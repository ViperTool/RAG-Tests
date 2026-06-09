import { Button } from "@/components/ui/button"
import { siteConfig } from "@/config/site"
import { Atom, Sparkles, Users, Zap } from "lucide-react"

export default function HeroBanner() {
  return (
    <div className="relative overflow-hidden bg-gradient-to-br from-blue-950 via-slate-900 to-slate-950 border-b border-border/40">
      {/* Animated background */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 rounded-full bg-blue-500/10 blur-3xl" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 rounded-full bg-blue-600/10 blur-3xl" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 rounded-full bg-blue-400/5 blur-3xl" />
      </div>

      <div className="container relative px-4 py-12 sm:py-16">
        <div className="flex flex-col items-center text-center space-y-6">
          {/* Logo animation */}
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-blue-700 rounded-full blur-xl opacity-50 animate-pulse" />
            <div className="relative flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-blue-700 shadow-2xl shadow-blue-500/30">
              <Atom className="h-10 w-10 text-white" />
            </div>
          </div>

          <div className="space-y-3 max-w-2xl">
            <h1 className="text-3xl sm:text-4xl font-bold tracking-tight">
              <span className="bg-gradient-to-r from-blue-400 via-blue-500 to-blue-600 bg-clip-text text-transparent">
                {siteConfig.title}
              </span>
            </h1>
            <p className="text-lg text-muted-foreground">
              Интеллектуальный помощник для работы с документами и знаниями
            </p>
          </div>

          {/* Feature badges */}
          <div className="flex flex-wrap justify-center gap-3">
            <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400">
              <Sparkles className="h-4 w-4" />
              <span className="text-sm font-medium">AI-помощник</span>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400">
              <Users className="h-4 w-4" />
              <span className="text-sm font-medium">Быстрые ответы</span>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400">
              <Zap className="h-4 w-4" />
              <span className="text-sm font-medium">Точный поиск</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}