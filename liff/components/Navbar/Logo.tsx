import React from 'react'
import Badminton from './Badminton SVG Icon.svg'
import Image from 'next/image'
import Link from 'next/link'

const Logo = () => {
  return (
    <div className="inline-flex h-[45px] justify-center p-3 align-middle text-lg">
      <Link href="/" className="inline-flex">
        <Image
          width={30}
          height={30}
          className="duration-150 hover:rotate-[-20deg]"
          src={Badminton}
          alt="Badminton Logo"
        />
        <p className="text-lg font-bold">Badminton</p>
      </Link>
    </div>
  )
}

export default Logo
