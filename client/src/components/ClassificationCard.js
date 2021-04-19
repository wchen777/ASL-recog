import React, { useEffect, useRef } from 'react'
import {
  Box, Badge
} from '@chakra-ui/react';

export default function ClassificationCard({ card }) {
  const cvRef = useRef()

  useEffect(() => {
    const c = cvRef.current
    const ctx = c.getContext('2d')

    ctx.drawImage(card.pic, 0, 0, c.width, c.height)

  }, [])

  return (
    <Box borderWidth="1px" borderRadius="lg" overflow="hidden" minWidth="200px">
      <canvas ref={cvRef} width="200" height="200">
      </canvas>
      <Badge borderRadius="full" px="2" colorScheme="blue" fontSize="xl" my={3}>
        {card.classification}
      </Badge>
    </Box>
  )
}
