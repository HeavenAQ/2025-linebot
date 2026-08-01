import React from 'react'
import Logo from './Logo'
import DropDownIcon from './DropdownIcon'

const Navbar = () => {
  return (
    <header className="fixed inset-x-0 top-0 z-30 border-b border-border/70 bg-background/85 backdrop-blur-md">
      <div className="mx-auto flex h-14 w-full max-w-content items-center justify-between px-4">
        <Logo />
        <DropDownIcon />
      </div>
    </header>
  )
}

export default Navbar
