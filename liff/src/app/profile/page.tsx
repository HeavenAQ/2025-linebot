'use client'

import { useEffect, useState } from 'react'

import ProfileForm from '@/components/ProfileForm'
import { Alert } from '@/components/ui/alert'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { PageContainer } from '@/components/ui/page'
import Spinner from '@/components/ui/spinner'
import Toast from '@/components/ui/toast'
import { fetchUserDataSafe } from '@/lib/api/fetchUserDataSafe'
import { useLiff } from '../LiffProvider'
import type { UserData } from '@/types'

/** 編輯個人資料: the details a student gave when they registered. */
export default function ProfilePage() {
  const { profile } = useLiff()
  const [user, setUser] = useState<UserData | null>(null)
  const [loadError, setLoadError] = useState('')
  const [loading, setLoading] = useState(true)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    if (!profile?.userId) return
    let active = true
    fetchUserDataSafe(profile.userId).then(result => {
      if (!active) return
      if (result.ok) setUser(result.data)
      else setLoadError(result.error.message)
      setLoading(false)
    })
    return () => {
      active = false
    }
  }, [profile?.userId])

  if (loading) return <Spinner fullscreen />

  return (
    <PageContainer className="pt-6">
      <Card>
        <CardHeader>
          <CardTitle>個人資料</CardTitle>
        </CardHeader>
        <CardContent>
          {loadError ? (
            <Alert variant="info" title="暫時無法載入">
              {loadError}
            </Alert>
          ) : (
            <ProfileForm
              user={user}
              submitLabel="儲存變更"
              onSaved={updated => {
                setUser(updated)
                setSaved(true)
              }}
            />
          )}
        </CardContent>
      </Card>
      {saved && <Toast message="已儲存個人資料" onDismiss={() => setSaved(false)} />}
    </PageContainer>
  )
}
