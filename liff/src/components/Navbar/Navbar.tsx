import React from 'react'
import Logo from './Logo'
import { mPlusRounded1c } from '../Fonts/M_PLUS_Rounded_1c'
import DropDownIcon from './DropdownIcon'
import ThemeToggleIcon from './ThemeToggleIcon'

const Navbar = () => {
  return (
    <div
      className={`${mPlusRounded1c.className} fixed left-1/2 z-20 flex h-12 w-full max-w-[800px] -translate-x-1/2 items-center justify-between p-2 backdrop-blur-md`}
    >
      <Logo />
      <div className="w-40px flex items-center justify-center gap-2">
        <ThemeToggleIcon />
        <DropDownIcon />
      </div>
    </div>
  )
}

export default Navbar
