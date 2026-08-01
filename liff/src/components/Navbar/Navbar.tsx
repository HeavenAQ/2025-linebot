import React from 'react'
import Logo from './Logo'
import DropDownIcon from './DropdownIcon'
import ThemeToggleIcon from './ThemeToggleIcon'

const Navbar = () => {
  return (
    <header className="fixed inset-x-0 top-0 z-30 border-b border-border/60 bg-background/80 backdrop-blur-md">
      <div className="mx-auto flex h-14 w-full max-w-content items-center justify-between px-4">
        <Logo />
        <div className="flex items-center gap-2">
          <ThemeToggleIcon />
          <DropDownIcon />
        </div>
      </div>
    </header>
  )
}

export default Navbar
