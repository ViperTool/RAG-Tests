import { Button } from "@/components/ui/button"
import { siteConfig } from "@/config/site"
import { Lightbulb } from "lucide-react"

export default function Examples({ onSelect, disabled }) {
  if (!siteConfig.ui.showExamples) return null

  const examplesToShow = siteConfig.examples.slice(0, siteConfig.ui.maxExamplesToShow)

  return (
    <div className="border-b border-border/40 bg-muted/30 p-4">
      <div className="container max-w-4xl mx-auto space-y-3">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Lightbulb className="h-4 w-4 text-yellow-500" />
          <span>Попробуйте спросить:</span>
        </div>
        <div className="flex flex-wrap gap-2">
          {examplesToShow.map((example, idx) => (
            <Button
              key={idx}
              variant="outline"
              size="sm"
              onClick={() => onSelect(example)}
              disabled={disabled}
              className="rounded-full text-xs sm:text-sm hover:bg-blue-500/10 hover:border-blue-500/30 hover:text-blue-400 transition-all"
            >
              {example}
            </Button>
          ))}
        </div>
      </div>
    </div>
  )
}