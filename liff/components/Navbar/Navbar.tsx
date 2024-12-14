import React from 'react'
import Logo from './Logo'
import { mPlusRounded1c } from '../Fonts/M_PLUS_Rounded_1c'
import DropDownIcon from './DropdownIcon'

const Navbar = () => {
  return (
    <div className={`${mPlusRounded1c.className} flex h-12 items-center justify-between p-2`}>
      <Logo />
      <DropDownIcon />
    </div>
  )
}

export default Navbar
