import clsx from 'clsx'
import { Bot, FileBarChart, Lightbulb, Moon, Package, Sun } from 'lucide-react'
import { NavLink, Route, Routes } from 'react-router'

import { BundlePage } from './features/bundles/BundlePage'
import { BundlesPage } from './features/bundles/BundlesPage'
import { PoliciesPage } from './features/policies/PoliciesPage'
import { PolicyPage } from './features/policies/PolicyPage'
import { PolicyResultPage } from './features/policies/PolicyResults'
import { ResultPage } from './features/results/ResultPage'
import { ResultsPage } from './features/results/ResultsPage'
import { useTheme } from './lib/themeContext'

/** 左の欄の移動先 */
function NavItem({ to, icon: Icon, label }: { to: string; icon: typeof Package; label: string }) {
  return (
    <NavLink
      to={to}
      end={to === '/'}
      className={({ isActive }) =>
        clsx(
          'flex items-center gap-2 rounded-lg px-3 py-2 text-sm',
          isActive
            ? 'bg-accent-soft font-medium text-accent-text'
            : 'text-ink-2 hover:bg-surface-2',
        )
      }
    >
      <Icon size={16} aria-hidden />
      {label}
    </NavLink>
  )
}

export function App() {
  const { theme, toggle } = useTheme()
  return (
    <div className="flex h-full">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:rounded focus:bg-surface focus:px-3 focus:py-2"
      >
        本文へ移る
      </a>
      <nav
        aria-label="移動"
        className="flex w-56 shrink-0 flex-col gap-1 border-r border-line bg-surface p-3"
      >
        <div className="mb-3 flex items-center justify-between px-2">
          <span className="flex items-center gap-2 text-base font-semibold">
            <Lightbulb size={20} className="text-accent" aria-hidden />
            ML-Provide
          </span>
          <button
            type="button"
            onClick={toggle}
            aria-label={theme === 'dark' ? '明るい表示にする' : '暗い表示にする'}
            className="rounded p-1 text-ink-3 hover:bg-surface-2"
          >
            {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>
        <NavItem to="/" icon={Package} label="推論の一式" />
        <NavItem to="/results" icon={FileBarChart} label="結果" />
        <NavItem to="/policies" icon={Bot} label="方策の一式" />
      </nav>
      <main id="main" className="min-w-0 flex-1 overflow-auto px-8 py-6">
        <Routes>
          <Route path="/" element={<BundlesPage />} />
          <Route path="/bundles/:id" element={<BundlePage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/results/:id" element={<ResultPage />} />
          <Route path="/policies" element={<PoliciesPage />} />
          <Route path="/policies/:id" element={<PolicyPage />} />
          <Route path="/policy-results/:id" element={<PolicyResultPage />} />
          <Route path="*" element={<p className="text-ink-3">ページが見つかりません.</p>} />
        </Routes>
      </main>
    </div>
  )
}
