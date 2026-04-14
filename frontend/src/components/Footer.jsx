import { siteConfig } from "@/config/site"
import { Separator } from "@/components/ui/separator"

export default function Footer() {
  return (
    <footer className="border-t border-border/40 bg-background/50">
      <div className="container max-w-4xl mx-auto px-4 py-4">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-muted-foreground">
          <p>{siteConfig.shortTitle} • v{siteConfig.version}</p>
          <p>Ответы генерируются на основе загруженных данных</p>
        </div>
      </div>
    </footer>
  )
}