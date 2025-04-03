(define (domain taxi_simplest)
  (:requirements :strips :typing)
  
  (:types
    person - object
    location - object
    taxi - object
  )
  
  (:predicates
    (plocation ?p - person ?l - location)
    (tlocation ?t - taxi ?l - location)
    (outsidetaxi ?p - person)
    (insidetaxi ?p - person ?t - taxi)
    (connects ?from - location ?to - location)
  )
  
  (:action get-in
    :parameters (?p - person ?t - taxi ?l - location)
    :precondition (and
                    (plocation ?p ?l)
                    (tlocation ?t ?l)
                    (outsidetaxi ?p)
                  )
    :effect (and
              (insidetaxi ?p ?t)
              (not (outsidetaxi ?p))
            )
  )
  
  (:action get-out
    :parameters (?p - person ?t - taxi ?l - location)
    :precondition (and
                    (insidetaxi ?p ?t)
                    (tlocation ?t ?l)
                  )
    :effect (and
              (plocation ?p ?l)
              (outsidetaxi ?p)
              (not (insidetaxi ?p ?t))
            )
  )
  
  (:action move-taxi
    :parameters (?t - taxi ?from - location ?to - location)
    :precondition (and
                    (tlocation ?t ?from)
                    (connects ?from ?to)
                  )
    :effect (and
              (not (tlocation ?t ?from))
              (tlocation ?t ?to)
            )
  )
)

